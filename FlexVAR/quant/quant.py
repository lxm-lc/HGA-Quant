import math
import os
from collections import Counter
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_hadamard_transform import hadamard_transform
from fast_hadamard_transform.fast_hadamard_transform_interface import (
    hadamard_transform_12N,
    hadamard_transform_20N,
    hadamard_transform_28N,
    hadamard_transform_40N,
)

try:
    from fast_hadamard_transform.fast_hadamard_transform_interface import hadamard_transform_rht
except (ImportError, AttributeError):
    hadamard_transform_rht = None


# Match the block-local Linear layers that correspond to the original
# attention / FFN quantization targets. Keep the embedding / output head path
# in floating point; those layers are disproportionately sensitive under
# hadamard+T-BGR fake quantization and were the source of visible banding.
LINEAR_QUANT_PREFIXES = (
    'blocks',
)


def _matches_prefix(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes)


def collect_quantizable_linear_modules(
    model: nn.Module,
    prefixes: Sequence[str] = LINEAR_QUANT_PREFIXES,
):
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _matches_prefix(name, prefixes)
    ]


class UniformQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int,
        token_wise: bool = False,
        symmetric: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        if not (2 <= n_bits <= 8):
            raise ValueError(f'n_bits must be in [2, 8], got {n_bits}')
        self.n_bits = int(n_bits)
        self.token_wise = bool(token_wise)
        self.symmetric = bool(symmetric)
        self.eps = float(eps)

        if self.symmetric:
            self.qmin = -(1 << (self.n_bits - 1))
            self.qmax = (1 << (self.n_bits - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = (1 << self.n_bits) - 1

        self.enabled = False

    def extra_repr(self):
        return (
            f'n_bits={self.n_bits}, token_wise={self.token_wise}, '
            f'symmetric={self.symmetric}, enabled={self.enabled}'
        )

    def set_bits(self, n_bits: int):
        if not (2 <= n_bits <= 8):
            raise ValueError(f'n_bits must be in [2, 8], got {n_bits}')
        self.n_bits = int(n_bits)
        if self.symmetric:
            self.qmin = -(1 << (self.n_bits - 1))
            self.qmax = (1 << (self.n_bits - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = (1 << self.n_bits) - 1

    def _compute_dynamic_params(self, x: torch.Tensor):
        x_new = x.detach().float()
        x_min = x_new.amin(dim=-1, keepdim=True) if self.token_wise else x_new.amin()
        x_max = x_new.amax(dim=-1, keepdim=True) if self.token_wise else x_new.amax()

        if self.symmetric:
            max_abs = torch.maximum(x_min.abs(), x_max.abs()).clamp(min=self.eps)
            scale = max_abs / max(self.qmax, 1)
            zero_point = torch.zeros_like(scale)
        else:
            scale = (x_max - x_min).clamp(min=self.eps) / max(self.qmax - self.qmin, 1)
            zero_point = torch.round(self.qmin - x_min / scale).clamp(self.qmin, self.qmax)

        return scale, zero_point

    def forward(self, x: torch.Tensor):
        if not self.enabled:
            return x

        scale, zero_point = self._compute_dynamic_params(x)

        x_int = torch.round(x / scale) + zero_point
        x_q = torch.clamp(x_int, self.qmin, self.qmax)
        x_quant = (x_q - zero_point) * scale
        return x_quant.to(dtype=x.dtype) if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else x_quant


class QuantLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        act_n_bits: int = 6,
        weight_n_bits: int = 4,
        act_token_wise: bool = True,
        act_symmetric: bool = False,
        quant_weight: bool = True,
        quant_act: bool = False,
        use_hadamard: bool = False,
        use_tbgr: bool = False,
        tbgr_outlier_factor: float = 8.0,
        tbgr_outlier_sigma: float = 3.0,
        hadamard_seed: Optional[int] = None,
    ):
        super().__init__()

        if not isinstance(linear, nn.Linear):
            raise TypeError(f'Expected nn.Linear, got {type(linear)}')
        if not (2 <= weight_n_bits <= 8):
            raise ValueError(f'weight_n_bits must be in [2, 8], got {weight_n_bits}')
        if not (2 <= act_n_bits <= 8):
            raise ValueError(f'act_n_bits must be in [2, 8], got {act_n_bits}')

        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.weight_n_bits = int(weight_n_bits)
        self.act_n_bits = int(act_n_bits)
        self.use_act_quant = bool(quant_act)
        self.use_weight_quant = bool(quant_weight)
        self.use_hadamard = bool(use_hadamard)
        self.use_tbgr = bool(use_tbgr)
        self.tbgr_outlier_factor = float(tbgr_outlier_factor)
        self.tbgr_outlier_sigma = float(tbgr_outlier_sigma)
        self.use_fused_rht = os.environ.get('INFINITY_USE_FUSED_RHT', '0').lower() in {'1', 'true', 'yes', 'on'}

        source_weight = linear.weight.detach().to(dtype=torch.float32)

        if hadamard_seed is None:
            rht_D = 2 * torch.randint(0, 2, (self.in_features,), dtype=torch.int8) - 1
        else:
            generator = torch.Generator()
            generator.manual_seed(int(hadamard_seed))
            rht_D = 2 * torch.randint(0, 2, (self.in_features,), dtype=torch.int8, generator=generator) - 1
        self.register_buffer('rht_D', rht_D, persistent=True)
        self.register_buffer('_rht_D_cache', None, persistent=False)
        self.register_buffer('_rht_D_int8_cache', None, persistent=False)

        if self.use_hadamard:
            source_weight = self._apply_random_hadamard(source_weight)

        if self.use_weight_quant:
            qweight, scale, zero_point = self._quantize_weight_tensor(
                source_weight,
                n_bits=self.weight_n_bits,
                use_tbgr=self.use_tbgr,
                tbgr_outlier_factor=self.tbgr_outlier_factor,
                tbgr_outlier_sigma=self.tbgr_outlier_sigma,
            )
            self.register_buffer('qweight', qweight, persistent=True)
            self.register_buffer('scale', scale.squeeze(1), persistent=True)
            self.register_buffer(
                'zero_point',
                None if zero_point is None else zero_point.squeeze(1),
                persistent=True,
            )
            self.register_buffer('fp_weight', None, persistent=False)
        else:
            self.register_buffer('qweight', None, persistent=True)
            self.register_buffer('scale', None, persistent=True)
            self.register_buffer('zero_point', None, persistent=True)
            self.register_buffer('fp_weight', source_weight, persistent=True)

        if linear.bias is None:
            self.register_buffer('bias', None, persistent=True)
        else:
            self.register_buffer('bias', linear.bias.detach().to(dtype=torch.float32), persistent=True)

        self.act_quantizer = UniformQuantizer(
            n_bits=self.act_n_bits,
            token_wise=act_token_wise,
            symmetric=act_symmetric,
        )
        self.act_quantizer.enabled = self.use_act_quant

    def extra_repr(self):
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'weight_n_bits={self.weight_n_bits}, act_n_bits={self.act_n_bits}, '
            f'use_act_quant={self.use_act_quant}, use_weight_quant={self.use_weight_quant}, '
            f'use_hadamard={self.use_hadamard}, use_tbgr={self.use_tbgr}'
        )

    def _get_rht_D(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if (
            self._rht_D_cache is None
            or self._rht_D_cache.device != device
            or self._rht_D_cache.dtype != dtype
        ):
            self._rht_D_cache = self.rht_D.to(device=device, dtype=dtype, non_blocking=True)
        return self._rht_D_cache

    def _get_rht_D_int8(self, device: torch.device) -> torch.Tensor:
        if self.rht_D.device == device and self.rht_D.dtype == torch.int8 and self.rht_D.is_contiguous():
            return self.rht_D
        if (
            self._rht_D_int8_cache is None
            or self._rht_D_int8_cache.device != device
            or self._rht_D_int8_cache.dtype != torch.int8
        ):
            self._rht_D_int8_cache = self.rht_D.to(device=device, dtype=torch.int8, non_blocking=True).contiguous()
        return self._rht_D_int8_cache

    def _apply_random_hadamard(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.shape[-1]
        if (
            self.use_fused_rht
            and hadamard_transform_rht is not None
            and x.is_cuda
            and dim > 0
            and (dim & (dim - 1)) == 0
        ):
            D_int8 = self._get_rht_D_int8(device=x.device)
            return hadamard_transform_rht(x, D_int8, scale=1.0 / math.sqrt(dim))

        D = self._get_rht_D(device=x.device, dtype=x.dtype)
        x = x * D
        if dim > 0 and (dim & (dim - 1)) == 0:
            return hadamard_transform(x, scale=1.0 / math.sqrt(dim))
        if dim % 12 == 0 and (dim // 12) > 0 and ((dim // 12) & ((dim // 12) - 1)) == 0:
            return hadamard_transform_12N(x, scale=1.0 / math.sqrt(dim))
        if dim % 20 == 0 and (dim // 20) > 0 and ((dim // 20) & ((dim // 20) - 1)) == 0:
            return hadamard_transform_20N(x, scale=1.0 / math.sqrt(dim))
        if dim % 28 == 0 and (dim // 28) > 0 and ((dim // 28) & ((dim // 28) - 1)) == 0:
            return hadamard_transform_28N(x, scale=1.0 / math.sqrt(dim))
        if dim % 40 == 0 and (dim // 40) > 0 and ((dim // 40) & ((dim // 40) - 1)) == 0:
            return hadamard_transform_40N(x, scale=1.0 / math.sqrt(dim))
        return x

    def _quantize_weight_tensor(
        self,
        weight: torch.Tensor,
        n_bits: int,
        use_tbgr: bool = False,
        tbgr_outlier_factor: float = 8.0,
        tbgr_outlier_sigma: float = 3.0,
    ):
        if use_tbgr:
            return T_BGR(
                weight,
                n_bits,
                num=100,
                outlier_factor=tbgr_outlier_factor,
                outlier_sigma=tbgr_outlier_sigma,
            )

        qmax = (1 << (n_bits - 1)) - 1
        qmin = -(1 << (n_bits - 1))
        max_abs = weight.abs().amax(dim=1, keepdim=True)
        scale = (max_abs / qmax).clamp(min=1e-8)
        qweight = torch.round(weight / scale).clamp(qmin, qmax).to(dtype=torch.int8)
        return qweight, scale, None

    def set_activation_bits(self, n_bits: int, enabled: bool = True):
        n_bits = int(n_bits)
        enabled = bool(enabled)
        if (
            self.act_n_bits == n_bits
            and self.use_act_quant == enabled
            and self.act_quantizer.enabled == enabled
            and self.act_quantizer.n_bits == n_bits
        ):
            return
        self.act_n_bits = n_bits
        self.use_act_quant = enabled
        self.act_quantizer.set_bits(n_bits)
        self.act_quantizer.enabled = enabled

    def _resolve_bias(self, x: torch.Tensor, bias: torch.Tensor = None):
        if bias is None:
            return self.bias
        return bias.to(device=x.device, dtype=torch.float32)

    def _materialize_weight(self, device: Optional[torch.device] = None) -> torch.Tensor:
        if self.use_weight_quant:
            if self.qweight is None or self.scale is None:
                raise ValueError('QuantLinear has no quantized weight buffer')

            if device is None:
                device = self.qweight.device
            return self._dequantize_weight(device=device)

        if self.fp_weight is None:
            raise ValueError('QuantLinear has no floating-point weight buffer')
        if device is not None:
            return self.fp_weight.to(device=device, dtype=torch.float32)
        return self.fp_weight

    @property
    def weight(self) -> torch.Tensor:
        return self._materialize_weight()

    def _dequantize_weight(self, device: torch.device) -> torch.Tensor:
        weight = self.qweight.to(device=device, dtype=torch.float32)
        scale = self.scale.to(device=device, dtype=torch.float32)
        if self.zero_point is None:
            return weight * scale.view(-1, 1)

        zero_point = self.zero_point.to(device=device, dtype=torch.float32)
        if scale.ndim == 1:
            scale = scale.view(-1, 1)
        if zero_point.ndim == 1:
            zero_point = zero_point.view(-1, 1)
        return (weight - zero_point) * scale

    def forward(self, x: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        if self.use_hadamard:
            x = self._apply_random_hadamard(x)

        if self.use_act_quant:
            x = self.act_quantizer(x)

        weight = self._materialize_weight(device=x.device)

        if x.dtype in (torch.float16, torch.bfloat16, torch.float32):
            weight = weight.to(dtype=x.dtype)

        bias = self._resolve_bias(x, bias)
        if bias is not None:
            bias = bias.to(device=x.device, dtype=weight.dtype)

        return F.linear(x, weight=weight, bias=bias)

    def forward_with_bias(self, x: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        return self.forward(x, bias=bias)


def linear_with_external_bias(linear: nn.Module, x: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    if hasattr(linear, 'forward_with_bias'):
        return linear.forward_with_bias(x, bias=bias)

    weight = linear.weight
    if bias is None:
        bias = linear.bias
    else:
        bias = bias.to(device=x.device)
    return F.linear(x, weight=weight, bias=bias)


def apply_linear_quantization(
    model: nn.Module,
    weight_quant: bool,
    act_quant: bool,
    weight_bits: int = 4,
    act_bits: int = 6,
    ctf_low_bits: int = 4,
    use_hadamard: bool = False,
    use_tbgr: bool = False,
    mixed_act_mode: str = 'default',
    tbgr_outlier_factor: float = 8.0,
    tbgr_outlier_sigma: float = 3.0,
    hadamard_seed: Optional[int] = None,
    quant_module_prefixes: Sequence[str] = LINEAR_QUANT_PREFIXES,
):
    if mixed_act_mode not in {'default', 'coarse_to_fine'}:
        raise ValueError(f'Unsupported mixed_act_mode={mixed_act_mode}')

    if mixed_act_mode == 'coarse_to_fine' and not (weight_quant and act_quant):
        raise ValueError('coarse_to_fine mode expects both --quantize_weight 1 and --quantize_act 1')

    target_modules = collect_quantizable_linear_modules(model, prefixes=quant_module_prefixes)

    replaced_module_names = []
    init_act_bits_by_module = {}

    for name, module in target_modules:
        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)
        else:
            child_name = name
            parent_module = model

        quant_module = QuantLinear(
            module,
            quant_weight=weight_quant,
            weight_n_bits=weight_bits,
            quant_act=act_quant,
            act_n_bits=ctf_low_bits if mixed_act_mode == 'coarse_to_fine' else act_bits,
            act_token_wise=True,
            use_hadamard=use_hadamard,
            use_tbgr=use_tbgr,
            tbgr_outlier_factor=tbgr_outlier_factor,
            tbgr_outlier_sigma=tbgr_outlier_sigma,
            hadamard_seed=hadamard_seed,
        )
        quant_module.train(module.training)
        setattr(parent_module, child_name, quant_module)

        replaced_module_names.append(name)
        init_act_bits_by_module[name] = (
            ctf_low_bits if mixed_act_mode == 'coarse_to_fine' else act_bits
        )

    init_act_bit_counter = Counter(init_act_bits_by_module.values())
    if init_act_bits_by_module:
        init_avg_act_bits = sum(
            bits * count for bits, count in init_act_bit_counter.items()
        ) / len(init_act_bits_by_module)
    else:
        init_avg_act_bits = 0.0

    return {
        'replaced_module_names': replaced_module_names,
        'num_replaced_modules': len(replaced_module_names),
        'weight_quant': weight_quant,
        'weight_bits': weight_bits,
        'act_quant': act_quant,
        'default_act_bits': act_bits,
        'mixed_act_mode': mixed_act_mode,
        'init_act_bits_by_module': init_act_bits_by_module,
        'init_act_bit_counter': dict(init_act_bit_counter),
        'init_avg_act_bits': init_avg_act_bits,
        'use_hadamard': use_hadamard,
        'use_tbgr': use_tbgr,
        'tbgr_outlier_factor': tbgr_outlier_factor,
        'tbgr_outlier_sigma': tbgr_outlier_sigma,
        'hadamard_seed': hadamard_seed,
        'quant_module_prefixes': tuple(quant_module_prefixes),
    }

def T_BGR(
    input: torch.Tensor,
    w_bit: int,
    num: int = 100,
    eps: float = 1e-6,
    max_iter: int = 8,
    outlier_factor: float = 8.0,
    outlier_sigma: float = 3.0,
):
    input = input.detach().to(dtype=torch.float32)

    if outlier_sigma <= 0:
        raise ValueError(f'outlier_sigma must be > 0, got {outlier_sigma}')

    channel_mean = input.mean(dim=1, keepdim=True)
    channel_std = input.std(dim=1, unbiased=False, keepdim=True).clamp(min=eps)
    outlier_mask = (input - channel_mean).abs() > (outlier_sigma * channel_std)

    err_weight = torch.ones_like(input)
    if outlier_factor != 1.0:
        err_weight = torch.where(
            outlier_mask,
            torch.full_like(input, float(outlier_factor)),
            err_weight,
        )
    err_weight_sum = err_weight.sum(dim=1).clamp(min=eps)

    qmax = (1 << w_bit) - 1
    q_dtype = torch.int16 if w_bit == 8 else torch.int8
    min_value = input.amin(dim=1)
    max_value = input.amax(dim=1)

    diff = (max_value - min_value) / (2 * num)
    best_err = torch.full_like(min_value, float('inf'))
    best_scale = torch.ones_like(min_value)
    best_zero_point = torch.zeros_like(min_value)
    best_qweight = torch.zeros_like(input, dtype=q_dtype)

    for i in range(num):
        lb = min_value + i * diff
        rb = max_value - i * diff
        clipped = torch.max(torch.min(input, rb.unsqueeze(-1)), lb.unsqueeze(-1))
        scale = ((clipped.amax(dim=1) - clipped.amin(dim=1)) / qmax).clamp(min=1e-8)
        zero_point = torch.round((-clipped.amin(dim=1) / scale)).clamp(0, qmax)
        qweight = torch.round(input / scale.unsqueeze(-1)) + zero_point.unsqueeze(-1)
        qweight = torch.clamp(qweight, 0, qmax)
        prev_err = None

        for _ in range(max_iter):
            centered = qweight - zero_point.unsqueeze(-1)
            denom = (err_weight * centered.square()).sum(dim=1).clamp(min=1e-8)
            scale = (err_weight * centered * input).sum(dim=1) / denom
            scale = scale.clamp(min=1e-8)
            zp_residual = qweight - input / scale.unsqueeze(-1)
            zero_point = torch.round(
                (err_weight * zp_residual).sum(dim=1) / err_weight_sum
            ).clamp(0, qmax)
            qweight = torch.round(input / scale.unsqueeze(-1)) + zero_point.unsqueeze(-1)
            qweight = torch.clamp(qweight, 0, qmax)
            dequant = (qweight - zero_point.unsqueeze(-1)) * scale.unsqueeze(-1)
            cur_err = (err_weight * (dequant - input).pow(2)).sum(dim=1) / err_weight_sum

            if prev_err is not None and torch.all((prev_err - cur_err).abs() < eps):
                break
            prev_err = cur_err

        dequant = (qweight - zero_point.unsqueeze(-1)) * scale.unsqueeze(-1)
        cur_err = (err_weight * (dequant - input).pow(2)).sum(dim=1) / err_weight_sum

        update_mask = cur_err < best_err
        best_err[update_mask] = cur_err[update_mask]
        best_scale[update_mask] = scale[update_mask]
        best_zero_point[update_mask] = zero_point[update_mask]
        best_qweight[update_mask] = qweight[update_mask].to(dtype=q_dtype)

    return best_qweight, best_scale.view(-1, 1), best_zero_point.view(-1, 1)

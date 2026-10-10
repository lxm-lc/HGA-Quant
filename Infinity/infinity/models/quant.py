import math
import os
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

LINEAR_QUANT_SUFFIXES = ('ffn.fc1', 'ffn.fc2', 'sa.proj', 'ca.mat_q', 'ca.proj', 'sa.mat_qkv', 'ca.mat_kv')

def _broadcast_token_like(param: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast a token-wise parameter across the token axis of ``x``."""
    if param is None:
        return None

    flat = param.reshape(-1)
    if x.ndim <= 1:
        return flat

    token_count = int(x.shape[-2])
    if flat.numel() == 1:
        values = flat.expand(token_count)
    elif flat.numel() == token_count:
        values = flat
    elif flat.numel() > token_count:
        values = flat[:token_count]
    elif flat.numel() == 2 and token_count > 1:
        sink = flat[:1]
        normal = flat[1:2]
        values = torch.cat((sink, normal.expand(token_count - 1)))
    else:
        tail = flat[-1:].expand(token_count - flat.numel())
        values = torch.cat((flat, tail))

    if x.ndim == 2:
        return values.view(-1, 1)
    return values.view(*([1] * (x.ndim - 2)), token_count, 1)


class UniformQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int,
        token_wise: bool = False,
        symmetric: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        if not (2 <= n_bits <= 8):
            raise ValueError(f'n_bits must be in [2, 8], got {n_bits}')
        self.n_bits = n_bits
        self.token_wise = token_wise
        self.symmetric = symmetric
        self.momentum = momentum
        self.eps = eps
        if symmetric:
            self.qmin = -(1 << (n_bits - 1))
            self.qmax = (1 << (n_bits - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = (1 << n_bits) - 1

        self.enabled = False
        self.inited = False
        self.register_buffer('scale', None, persistent=True)
        self.register_buffer('zero_point', None, persistent=True)

    def extra_repr(self):
        return (
            f'n_bits={self.n_bits}, token_wise={self.token_wise}, '
            f'symmetric={self.symmetric}, enabled={self.enabled}, '
            f'inited={self.inited}'
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

    # def clear_static_params(self):
    #     self.scale = None
    #     self.zero_point = None
    #     self.inited = False

    # def load_static_params(self, scale: torch.Tensor, zero_point: torch.Tensor, enabled: bool = True):
    #     self.scale = torch.as_tensor(scale, dtype=torch.float32).detach().clone()
    #     self.zero_point = torch.as_tensor(zero_point, dtype=torch.float32).detach().clone()
    #     self.enabled = bool(enabled)
    #     self.inited = True
    #     return self

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        scale_key = prefix + 'scale'
        zero_point_key = prefix + 'zero_point'

        if scale_key in state_dict and self.scale is None:
            self.scale = state_dict[scale_key].detach().clone()
        if zero_point_key in state_dict and self.zero_point is None:
            self.zero_point = state_dict[zero_point_key].detach().clone()
        if self.scale is not None and self.zero_point is not None:
            self.inited = True
            self.enabled = True

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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

    def _resolve_static_params(self, x: torch.Tensor):
        if self.scale is None or self.zero_point is None:
            return self._compute_dynamic_params(x)

        scale = self.scale.to(device=x.device, dtype=torch.float32)
        zero_point = self.zero_point.to(device=x.device, dtype=torch.float32)
        if self.token_wise:
            scale = _broadcast_token_like(scale, x)
            zero_point = _broadcast_token_like(zero_point, x)
        elif x.ndim > 1:
            scale = scale.view(*([1] * (x.ndim - 1)), -1)
            zero_point = zero_point.view(*([1] * (x.ndim - 1)), -1)
        return scale.clamp(min=self.eps), zero_point

    def forward(self, x: torch.Tensor):
        if not self.enabled:
            return x

        if self.scale is not None and self.zero_point is not None:
            scale, zero_point = self._resolve_static_params(x)
        else:
            scale, zero_point = self._compute_dynamic_params(x)

        x_int = torch.round(x / scale) + zero_point
        x_q = torch.clamp(x_int, self.qmin, self.qmax)
        x_quant = (x_q - zero_point) * scale
        if x.dtype in (torch.float16, torch.bfloat16, torch.float32):
            return x_quant.to(dtype=x.dtype)
        return x_quant


class QuantLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        act_n_bits: int = 6,
        weight_n_bits: int = 4,
        act_token_wise: bool = False,
        act_symmetric: bool = False,
        quant_weight: bool = True,
        quant_act: bool = False,
        use_hadamard: bool = False,
        use_tbgr: bool = False,
        tbgr_outlier_factor: float = 8.0,
        tbgr_outlier_sigma: float = 3.0,
    ):
        super().__init__()

        if not isinstance(linear, nn.Linear):
            raise TypeError(f'Expected nn.Linear, got {type(linear)}')
        if not (2 <= weight_n_bits <= 8):
            raise ValueError(f'n_bits must be in [2, 8], got {weight_n_bits}')

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight_n_bits = weight_n_bits
        self.act_n_bits = act_n_bits
        self.use_act_quant = quant_act
        self.use_weight_quant = quant_weight
        
        self.use_hadamard = use_hadamard
        self.use_fused_rht = os.environ.get('INFINITY_USE_FUSED_RHT', '0').lower() in {'1', 'true', 'yes', 'on'}
        # This tensor consumes the global RNG state, which is fixed upstream via seed_everything
        self.register_buffer(
            'rht_D',
            2 * torch.randint(0, 2, (self.in_features,), dtype=torch.int8) - 1,
            persistent=True,
        )
        self.register_buffer('_rht_D_cache', None, persistent=False)
        self.register_buffer('_rht_D_int8_cache', None, persistent=False)
        self.use_tbgr = use_tbgr

        weight = linear.weight.detach().to(dtype=torch.float32)
        if self.use_hadamard:
            weight = self._apply_random_hadamard(weight)
        if self.use_weight_quant:
            qweight, scale, zero_point = self._quantize_weight_tensor(
                weight,
                n_bits=weight_n_bits,
                use_tbgr=use_tbgr,
                tbgr_outlier_factor=tbgr_outlier_factor,
                tbgr_outlier_sigma=tbgr_outlier_sigma,
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
            self.register_buffer('fp_weight', weight, persistent=True)
            
        if linear.bias is None:
            self.register_buffer('bias', None, persistent=True)
        else:
            self.register_buffer('bias', linear.bias.detach(), persistent=True)

        self.act_quantizer = UniformQuantizer(
            n_bits=act_n_bits,
            token_wise=act_token_wise,
            symmetric=act_symmetric,
        )
        self.act_quantizer.enabled = quant_act

    def extra_repr(self):
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'weight_n_bits={self.weight_n_bits}, act_n_bits={self.act_n_bits}, '
            f'use_act_quant={self.use_act_quant}, use_weight_quant={self.use_weight_quant}, '
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
            qweight, scale, zero_point = T_BGR(
                weight, 
                n_bits, 
                num=100,
                outlier_factor=tbgr_outlier_factor,
                outlier_sigma=tbgr_outlier_sigma,
            )
            return qweight, scale, zero_point

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

    def forward(self, x: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        if self.use_hadamard:
            x = self._apply_random_hadamard(x)
        
        output = self._fake_quant_forward(x, bias=bias)
        return output
    
    def _fake_quant_forward(self, x: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        if self.use_act_quant:
            x = self.act_quantizer(x)

        if self.use_weight_quant:
            weight = self._dequantize_weight(device=x.device)
        else:
            weight = self.fp_weight.to(device=x.device, dtype=torch.float32)
        if x.dtype in (torch.float16, torch.bfloat16, torch.float32):
            weight = weight.to(dtype=x.dtype)

        if bias is None:
            bias = None if self.bias is None else self.bias.to(device=x.device, dtype=weight.dtype)
        else:
            bias = bias.to(device=x.device, dtype=weight.dtype)

        return F.linear(x, weight=weight, bias=bias)

    def forward_with_bias(self, x: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        return self.forward(x, bias=bias)

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
    use_hadamard=False,
    use_tbgr=False,
    tbgr_outlier_factor: float = 8.0,
    tbgr_outlier_sigma: float = 3.0,
):
    if not weight_quant and not act_quant:
        return

    target_modules = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(suffix) for suffix in LINEAR_QUANT_SUFFIXES):
            continue
        target_modules.append((name, module))

    for name, module in target_modules:
        act_bits_this = act_bits
        
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
            act_n_bits=act_bits_this,
            act_token_wise=True,
            use_hadamard=use_hadamard,
            use_tbgr=use_tbgr,
            tbgr_outlier_factor=tbgr_outlier_factor,
            tbgr_outlier_sigma=tbgr_outlier_sigma,
        )
        setattr(parent_module, child_name, quant_module)


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
    # Asymmetric 8-bit weights span [0, 255], so int8 would overflow here.
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

        # cur_err = torch.mean(((qweight - zero_point.unsqueeze(-1)) * scale.unsqueeze(-1) - input) ** 2, dim=1)
        dequant = (qweight - zero_point.unsqueeze(-1)) * scale.unsqueeze(-1)
        cur_err = (err_weight * (dequant - input).pow(2)).sum(dim=1) / err_weight_sum
        
        update_mask = cur_err < best_err
        best_err[update_mask] = cur_err[update_mask]
        best_scale[update_mask] = scale[update_mask]
        best_zero_point[update_mask] = zero_point[update_mask]
        best_qweight[update_mask] = qweight[update_mask].to(dtype=q_dtype)

    return best_qweight, best_scale.view(-1, 1), best_zero_point.view(-1, 1)

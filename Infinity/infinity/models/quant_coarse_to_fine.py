import math
from collections import Counter
from infinity.models.quant import QuantLinear

def _token_count(pn):
    tokens = 1
    for value in pn:
        tokens *= int(value)
    return int(tokens)


# def _format_int(value):
#     return f'{int(value):,}'


def _stage_linear_flops(quant_modules, stage_tokens, batch_size=1, ca_kv_tokens=0):
    image_tokens = int(stage_tokens) * int(batch_size)
    text_tokens = int(ca_kv_tokens)

    total_flops = 0
    for name, module in quant_modules:
        tokens = text_tokens if name.endswith('ca.mat_kv') else image_tokens
        if tokens <= 0:
            continue
        total_flops += 2 * int(tokens) * int(module.in_features) * int(module.out_features)

    return int(total_flops)


# def _format_stage_plan(stage_plan):
#     return ', '.join(
#         f"s{row['stage_idx']:02d}:{row['bits']}b/"
#         f"{row['tokens']}tok/{_format_int(row['flops'])}flop"
#         for row in stage_plan
#     )


class CoarseToFineFlopsBudgetPolicy:
    def __init__(
        self,
        policy='early_high',
        low_bits=4,
        high_bits=8,
        target_avg_bits=6.0,
        u_late_high_count=1,
    ):
        self.policy = policy
        self.low_bits = int(low_bits)
        self.high_bits = int(high_bits)
        self.target_avg_bits = float(target_avg_bits)
        self.u_late_high_count = int(u_late_high_count)

        if self.high_bits <= self.low_bits:
            raise ValueError(
                f'ctf_high_bits must be greater than ctf_low_bits, got '
                f'{self.high_bits} <= {self.low_bits}.'
            )

        if self.target_avg_bits < self.low_bits:
            raise ValueError(
                f'ctf_target_avg_bits={self.target_avg_bits} is smaller than '
                f'ctf_low_bits={self.low_bits}; no feasible schedule exists.'
            )

    def build_plan(self, scale_schedule, quant_modules, batch_size=1, ca_kv_tokens=0):
        tokens_by_stage = [_token_count(pn) for pn in scale_schedule]
        flops_by_stage = [
            _stage_linear_flops(
                quant_modules,
                stage_tokens=tokens,
                batch_size=batch_size,
                ca_kv_tokens=ca_kv_tokens,
            )
            for tokens in tokens_by_stage
        ]

        total_tokens = sum(tokens_by_stage)
        total_flops = sum(flops_by_stage)
        high_flop_budget = self._high_flop_budget(total_flops)

        if self.policy == 'early_high':
            high_stages = self._select_early(flops_by_stage, high_flop_budget)

        elif self.policy == 'late_high':
            high_stages = self._select_late(flops_by_stage, high_flop_budget)

        elif self.policy == 'u_shape':
            high_stages = self._select_u_shape(flops_by_stage, high_flop_budget)

        else:
            raise ValueError(f'Unknown ctf policy: {self.policy}')

        stage_plan = []
        high_flops = 0
        weighted_flop_bits = 0

        for si, (pn, tokens, flops) in enumerate(zip(scale_schedule, tokens_by_stage, flops_by_stage)):
            bits = self.high_bits if si in high_stages else self.low_bits
            if si in high_stages:
                high_flops += flops
            weighted_flop_bits += flops * bits
            stage_plan.append(
                {
                    'stage_idx': si,
                    'pn': tuple(int(v) for v in pn),
                    'tokens': tokens,
                    'flops': flops,
                    'bits': bits,
                }
            )

        avg_bits = weighted_flop_bits / total_flops if total_flops else 0.0

        if avg_bits > self.target_avg_bits + 1e-12:
            raise RuntimeError(
                f'Internal budget error: flops_avg_bits={avg_bits:.6f} exceeds '
                f'target={self.target_avg_bits:.6f}.'
            )

        return {
            'policy': self.policy,
            'budget_metric': 'linear_flops',
            'low_bits': self.low_bits,
            'high_bits': self.high_bits,
            'target_avg_bits': self.target_avg_bits,
            'batch_size': int(batch_size),
            'ca_kv_tokens': int(ca_kv_tokens),
            'total_tokens': total_tokens,
            'total_flops': total_flops,
            'high_flop_budget': high_flop_budget,
            'used_high_flops': high_flops,
            'avg_bits': avg_bits,
            'high_stages': sorted(high_stages),
            'stage_plan': stage_plan,
        }

    def _high_flop_budget(self, total_flops):
        if total_flops <= 0:
            return 0

        extra_budget = (self.target_avg_bits - self.low_bits) * total_flops
        extra_per_high_flop = self.high_bits - self.low_bits

        if extra_budget < 0:
            return 0

        return int(math.floor(extra_budget / extra_per_high_flop + 1e-12))

    def _select_early(self, flops_by_stage, high_flop_budget):
        high_stages = set()
        used = 0

        for si, flops in enumerate(flops_by_stage):
            if used + flops > high_flop_budget:
                break
            high_stages.add(si)
            used += flops

        return high_stages

    def _select_late(self, flops_by_stage, high_flop_budget):
        high_stages = set()
        used = 0

        for si in range(len(flops_by_stage) - 1, -1, -1):
            flops = flops_by_stage[si]
            if used + flops > high_flop_budget:
                break
            high_stages.add(si)
            used += flops

        return high_stages

    def _select_u_shape(self, flops_by_stage, high_flop_budget):
        num_stages = len(flops_by_stage)

        if self.u_late_high_count <= 0:
            raise ValueError('--ctf_u_late_high_count must be positive for u_shape.')

        if self.u_late_high_count > num_stages:
            raise ValueError(
                f'ctf_u_late_high_count={self.u_late_high_count} exceeds '
                f'num_stages={num_stages}.'
            )

        late_stages = set(range(num_stages - self.u_late_high_count, num_stages))
        late_flops = sum(flops_by_stage[si] for si in late_stages)

        if late_flops > high_flop_budget:
            raise ValueError(
                'The requested late high-bit stages already exceed the FLOPs budget. '
                f'late_high_count={self.u_late_high_count}, '
                f'late_flops={late_flops}, high_flop_budget={high_flop_budget}, '
                f'low_bits={self.low_bits}, high_bits={self.high_bits}, '
                f'target_avg_bits={self.target_avg_bits}.'
            )

        high_stages = set(late_stages)
        used = late_flops

        for si, flops in enumerate(flops_by_stage):
            if si in late_stages:
                break
            if used + flops > high_flop_budget:
                break
            high_stages.add(si)
            used += flops

        return high_stages



class CoarseToFineActScheduler:
    def __init__(self, model, policy):
        self.policy = policy
        self.quant_modules = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, QuantLinear)
        ]

        self.current_plan = None
        self.current_key = None
        self.current_bits = None
        self.stage_bit_counter = Counter()

    def _set_all_bits(self, bits: int):
        bits = int(bits)
        if self.current_bits == bits:
            return
        for _, module in self.quant_modules:
            if module.act_n_bits != bits or not module.use_act_quant:
                module.set_activation_bits(bits, enabled=True)
        self.current_bits = bits
    
    def begin(self, scale_schedule, batch_size=1, ca_kv_tokens=0):
        key = (
            tuple(tuple(int(v) for v in pn) for pn in scale_schedule),
            int(batch_size),
            int(ca_kv_tokens),
        )

        if key != self.current_key:
            self.current_plan = self.policy.build_plan(
                scale_schedule,
                self.quant_modules,
                batch_size=batch_size,
                ca_kv_tokens=ca_kv_tokens,
            )
            self.current_key = key

        self.stage_bit_counter = Counter()

    def apply_prefill(self):
        if self.current_plan is None:
            raise RuntimeError('CoarseToFineActScheduler.begin() must be called first.')
        self._set_all_bits(self.current_plan['high_bits'])

    def apply_stage(self, si):
        if self.current_plan is None:
            raise RuntimeError('CoarseToFineActScheduler.begin() must be called first.')

        bits = self.current_plan['stage_plan'][si]['bits']
        self._set_all_bits(bits)
        self.stage_bit_counter[bits] += 1

    def summary(self):
        if self.current_plan is None:
            return {}
        return {
            **self.current_plan,
            'num_quant_modules': len(self.quant_modules),
            'stage_bit_counter': dict(self.stage_bit_counter),
        }


def build_coarse_to_fine_act_scheduler(model, args):
    policy = CoarseToFineFlopsBudgetPolicy(
        policy=getattr(args, 'ctf_act_policy', 'early_high'),
        low_bits=getattr(args, 'ctf_low_bits', 4),
        high_bits=getattr(args, 'ctf_high_bits', 8),
        target_avg_bits=getattr(args, 'ctf_target_avg_bits', 6.0),
        u_late_high_count=getattr(args, 'ctf_u_late_high_count', 1),
    )

    return CoarseToFineActScheduler(model=model, policy=policy)

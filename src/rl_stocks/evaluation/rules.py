"""Fixed causal execution diagnostics; these are not learned policies."""
import math


class ExecutionRule:
    def __init__(self, rule, normalizer):
        if rule not in ('cash', 'long', 'short', 'ema_trend'):
            raise ValueError('unknown execution rule')
        self.rule = rule
        self.normalizer = normalizer
        self.indices = [normalizer.names.index(n) for n in ('ema_20_ratio', 'ema_50_ratio')]

    def predict(self, observation, deterministic=True):
        if len(observation) != len(self.normalizer.names) + 5:
            raise ValueError('observation schema mismatch')
        if not all(math.isfinite(float(v)) for v in observation):
            raise ValueError('nonfinite rule observation')
        exposure = {'cash': 0., 'long': .95, 'short': -.95}.get(self.rule, 0.)
        if self.rule == 'ema_trend':
            # Ratios are current close / causal EMA - 1. Comparing the two
            # reconstructs EMA ordering without seeing future prices or labels.
            fast, slow = [float(observation[i]) * self.normalizer.scales[i]
                          + self.normalizer.means[i] for i in self.indices]
            exposure = .95 if slow > fast > 0 else -.95 if slow < fast < 0 else 0.
        # Fixed wide protection is an execution diagnostic, not a learned zone.
        return [exposure, 1., 1., 1.], None

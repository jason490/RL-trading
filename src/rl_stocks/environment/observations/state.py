"""Versioned, causal portfolio/execution inputs shared by research and inference."""
import math
import numpy as np

PROFILES = ('legacy', 'execution_masked', 'execution_full')
LEGACY_FIELDS = ('exposure', 'cash_fraction', 'holding_age', 'unrealized_fraction', 'equity_return')
EXECUTION_FIELDS = ('stop_distance', 'near_distance', 'far_distance', 'protection_present',
                    'near_done', 'last_stop_action', 'last_near_action', 'last_width_action',
                    'unit_share_fraction')


def observation_spec(profile='legacy'):
    if profile not in PROFILES:
        raise ValueError('unknown observation profile')
    return {'version': 1, 'profile': profile,
            'portfolio_fields': list(LEGACY_FIELDS + (() if profile == 'legacy' else EXECUTION_FIELDS))}


def portfolio_state(sim, price, atr=None, profile='legacy'):
    observation_spec(profile)
    p, config = sim.portfolio, sim.config
    equity = p.equity(price)
    denominator = max(equity, 1e-8)
    # Preserve the original operation order, including float32 conversion downstream.
    values = [p.quantity*price/denominator, p.cash/denominator,
              p.age/config.max_holding, (price-p.basis)*p.quantity/denominator,
              equity/config.initial_equity-1]
    if profile == 'execution_masked':
        values += [0.] * len(EXECUTION_FIELDS)
    elif profile == 'execution_full':
        if not math.isfinite(price) or price <= 0 or atr is None or not math.isfinite(atr) or atr <= 0:
            raise ValueError('positive finite known price and ATR required')
        protected = bool(p.quantity and sim.protection is not None)
        direction = 1 if p.quantity > 0 else -1
        distances = ([float(np.clip(direction*(level-price)/atr, -20., 20.))/20.
                      for level in sim.protection] if protected else [0.] * 3)
        previous = list(sim.last_protective_action) if p.quantity and sim.last_protective_action is not None else [0.] * 3
        if len(distances) != 3 or len(previous) != 3 or any(abs(x) > 1 for x in previous):
            raise ValueError('invalid held execution state')
        values += distances + [float(protected), float(bool(p.quantity and sim.near_done))] + previous
        values += [float(np.clip(price/denominator, 0., 1.))]
    if not np.isfinite(values).all():
        raise ValueError('nonfinite portfolio state')
    return values

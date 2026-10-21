"""Universal sign restrictions for fixed-policy counterfactual diagnostics."""
import numpy as np

EXPOSURE_MODES = ('unrestricted', 'long_only', 'short_only', 'unit_magnitude')


def filter_exposure(action, mode):
    if mode not in EXPOSURE_MODES:
        raise ValueError('unknown exposure filter')
    result = np.array(action, copy=True)
    if result.shape != (4,) or not np.isfinite(result).all() or np.any(np.abs(result) > 1):
        raise ValueError('finite four-component bounded action required')
    if mode == 'long_only':
        result[0] = max(0, result[0])
    elif mode == 'short_only':
        result[0] = min(0, result[0])
    elif mode == 'unit_magnitude':
        result[0] = np.sign(result[0])
    return result

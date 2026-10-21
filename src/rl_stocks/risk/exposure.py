"""Shared exposure transforms for explicit training, evaluation and frozen execution contracts."""
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


EXECUTION_MODES = ('unrestricted', 'unit_magnitude')


def execution_contract(training='unrestricted', evaluation='unrestricted'):
    if training not in EXECUTION_MODES or evaluation not in EXECUTION_MODES:
        raise ValueError('unsupported training/evaluation exposure execution mode')
    return dict(version=1, training_exposure_mode=training, evaluation_exposure_mode=evaluation)


def validate_execution_contract(value):
    if not isinstance(value, dict) or type(value.get('version')) is not int or value != execution_contract(
            value.get('training_exposure_mode'), value.get('evaluation_exposure_mode')):
        raise ValueError('invalid exposure execution contract')
    return value

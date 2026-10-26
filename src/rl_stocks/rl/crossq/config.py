"""Only the preregistered CrossQ package is exposed through the factory."""

from rl_stocks.core.contracts import canonical
from rl_stocks.training.crossq_spec import parameters as study_parameters


def parameters(smoke=False):
    return study_parameters('crossq', smoke)


def validate_parameters(value, smoke=False):
    if canonical(value) != canonical(parameters(smoke)):
        raise ValueError('fixed preregistered CrossQ parameters required')

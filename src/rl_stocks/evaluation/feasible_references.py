"""Three ex-ante constant requests on fixed fitted history; no learning or selection."""
from dataclasses import dataclass, field
import json

import numpy as np

from rl_stocks.core.contracts import digest
from rl_stocks.evaluation.offpolicy_fit import ROOT

PROTOCOL_SHA256 = '67248a9cfa7c99852bdf4a88fe6d7c7584ab14a1ef3b6537b20df20d19d21051'
SOURCE_FILES = ('configs/experiments/feasible_references.json',
                'src/rl_stocks/evaluation/feasible_references.py',
                'scripts/diagnose_feasible_references.py',
                'src/rl_stocks/evaluation/offpolicy_fit.py',
                'src/rl_stocks/evaluation/training_fit.py',
                'src/rl_stocks/evaluation/runner.py',
                'src/rl_stocks/risk/decisions.py',
                'src/rl_stocks/environment/execution/simulator.py')


def protocol():
    value = json.loads((ROOT/SOURCE_FILES[0]).read_text())
    if digest(value) != PROTOCOL_SHA256:
        raise ValueError('fixed three-reference training-only protocol required')
    return value


@dataclass(frozen=True)
class FixedReference:
    name: str
    action: tuple = field(init=False)

    def __post_init__(self):
        actions = protocol()['actions']
        if self.name not in actions:
            raise ValueError('only fixed long/short/cash references allowed')
        object.__setattr__(self, 'action', tuple(actions[self.name]))

    def predict(self, observation, deterministic=True):
        if not deterministic:
            raise ValueError('deterministic reference required')
        return np.array(self.action, dtype=np.float32), None

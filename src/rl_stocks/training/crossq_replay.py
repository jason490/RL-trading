"""Read-only final replay/timeout acceptance; no replay sampling or learning."""

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

from rl_stocks.evaluation.soft_consistency import fingerprint
from rl_stocks.training.crossq_spec import protocol


def replay_record(buffer, transitions):
    if (type(buffer) is not ReplayBuffer or type(transitions) is not int
        or not 0 <= transitions <= protocol()['steps_per_cell']
        or buffer.buffer_size != 100000 or buffer.n_envs != 1
        or buffer.optimize_memory_usage is not False
        or buffer.handle_timeout_termination is not True
        or type(buffer.pos) is not int or buffer.pos != transitions
        or buffer.full is not False):
        raise ValueError('fixed single-env unwrapped replay/timeout geometry required')
    shapes = dict(observations=(transitions, 1, 21), next_observations=(transitions, 1, 21),
                  actions=(transitions, 1, 4), rewards=(transitions, 1),
                  dones=(transitions, 1), timeouts=(transitions, 1))
    hashes = {}
    for name, shape in shapes.items():
        values = getattr(buffer, name)[:transitions]
        if values.shape != shape or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError('finite float32 filled replay fields required')
        hashes[name] = fingerprint(values)
    if (not np.isin(buffer.dones[:transitions], [0, 1]).all()
        or not np.isin(buffer.timeouts[:transitions], [0, 1]).all()
        or bool((buffer.timeouts[:transitions] > buffer.dones[:transitions]).any())
        or bool((np.abs(buffer.actions[:transitions]) > 1).any())):
        raise ValueError('valid stored done/timeout masks and normalized actions required')
    return dict(buffer_size=100000, filled_transitions=transitions, n_envs=1,
                handle_timeout_termination=True, optimized_memory=False,
                done_count=int(buffer.dones[:transitions].sum()),
                terminal_count=int((buffer.dones[:transitions] - buffer.timeouts[:transitions]).sum()),
                timeout_count=int(buffer.timeouts[:transitions].sum()),
                filled_field_sha256=hashes, unfilled_capacity_ignored=True)

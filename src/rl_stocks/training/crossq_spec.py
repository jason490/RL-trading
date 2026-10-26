"""Fixed SAC/CrossQ comparison and phase-sensitive expected update geometry."""

from copy import deepcopy
import json
from pathlib import Path

from rl_stocks.core.contracts import digest

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SHA256 = '0d92529f39a9e18001add858e44b849db16d58174e71f86ee273dd0b21ffe61c'


def protocol():
    value = json.loads((ROOT / 'configs/experiments/crossq_comparison_preregistration.json').read_text())
    if digest(value) != PROTOCOL_SHA256:
        raise ValueError('fixed SAC/CrossQ preregistration required')
    return value


def parameters(arm, smoke=False):
    spec = protocol()
    if arm not in spec['arms'] or type(smoke) is not bool:
        raise ValueError('fixed CrossQ comparison arm/boolean phase required')
    result = deepcopy(spec['parameters'])
    if smoke:
        result.update(spec['smoke_parameters'])
    result.update(spec['arm_parameters'][arm])
    result['policy_kwargs'] = deepcopy(spec['policy_parameters'])
    result['policy_kwargs'].update(deepcopy(spec['arm_policy_parameters'][arm]))
    result['policy_kwargs']['optimizer_kwargs']['betas'] = tuple(result['policy_kwargs']['optimizer_kwargs']['betas'])
    return result


def expected_updates(arm, steps, phase, smoke=False):
    spec = protocol()
    params = parameters(arm, smoke)
    maximum = spec['smoke_steps' if smoke else 'steps_per_cell']
    if type(steps) is not int or not 0 <= steps <= maximum or phase not in ('on_step', 'post_update'):
        raise ValueError('bounded integer transition count and checkpoint phase required')
    critic = max(0, steps - params['learning_starts'] - int(phase == 'on_step'))
    actor = critic if arm == 'sac' else critic // params['policy_delay']
    result = dict(critic=critic, actor=actor, entropy=actor)
    if arm == 'crossq':
        warmup = params['policy_kwargs']['renorm_warmup_steps']
        result.update(critic_normalization=critic, actor_normalization=actor,
                      critic_renorm_forward_calls=max(0, critic - warmup - 1),
                      actor_renorm_forward_calls=max(0, actor - warmup - 1))
    return result

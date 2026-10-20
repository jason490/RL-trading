"""Pinned TD3 execution-state research contract; no fitting at import."""
import json
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.rl.registry.factory import resolved_parameters

PROTOCOL_SHA256 = 'c7cfeb51b279e910b71a9d80522486a570fb8e3be156b639e856c5f02a9ab32a'


def protocol():
    p = json.loads((Path(__file__).resolve().parents[3] / 'configs/experiments/td3_execution_state.json').read_text())
    if digest(p) != PROTOCOL_SHA256:
        raise ValueError('TD3 state protocol changed')
    actual = resolved_parameters('td3')
    actual['net_arch'] = actual.pop('policy_kwargs')['net_arch']
    if actual != p['parameters']:
        raise ValueError('TD3 parameters changed')
    return p


def validate_payload(data):
    p = protocol()
    if digest(data) != p['dataset']:
        raise ValueError('TD3 state dataset identity mismatch')
    if (data['manifest']['included_splits'] != ['train', 'validation']
            or data['manifest']['unverified_override'] is not False
            or any(e['split'] not in ('train', 'validation')
                   or any(b['session'] > '2023-12-31' for b in e['bars'])
                   for e in data['episodes'].values())):
        raise ValueError('TD3 state dataset contract mismatch')
    return data


def initialization_record(model):
    """Read initial tensors/noise without sampling actions or changing RNG state."""
    import hashlib
    import numpy as np
    from stable_baselines3.common.noise import NormalActionNoise
    if not isinstance(model.action_noise, NormalActionNoise):
        raise ValueError('TD3 Gaussian exploration required')
    hashes = {}
    for name in ('actor', 'actor_target', 'critic', 'critic_target'):
        h = hashlib.sha256()
        for key, value in sorted(getattr(model, name).state_dict().items()):
            h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
        hashes[name] = h.hexdigest()
    noise = model.action_noise
    if np.asarray(noise._mu).shape != (4,) or np.asarray(noise._sigma).shape != (4,):
        raise ValueError('TD3 four-action exploration required')
    record = dict(network_sha256=hashes, exploration_noise=dict(kind='NormalActionNoise',
                  mean=noise._mu.tolist(), sigma=noise._sigma.tolist()),
                  policy_delay=model.policy_delay, target_policy_noise=model.target_policy_noise,
                  target_noise_clip=model.target_noise_clip)
    validate_initialization(record)
    return record


def validate_initialization(record):
    import re
    p = protocol()
    if (set(record) != {'network_sha256', 'exploration_noise', 'policy_delay', 'target_policy_noise', 'target_noise_clip'}
            or record['exploration_noise'] != p['exploration_noise']
            or any(record[k] != p['parameters'][k] for k in ('policy_delay', 'target_policy_noise', 'target_noise_clip'))
            or set(record['network_sha256']) != {'actor', 'actor_target', 'critic', 'critic_target'}
            or any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v)
                   for v in record['network_sha256'].values())):
        raise ValueError('TD3 initialization/noise contract mismatch')
    return record

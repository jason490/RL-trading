"""Fixed critic study parameters and fresh off-policy initialization evidence."""
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.evaluation.soft_consistency import fingerprint
from rl_stocks.rl.registry.factory import resolved_parameters
from rl_stocks.rl.tqc.geometry import model_spec
from rl_stocks.training.sac_temperature import initialization_record
from rl_stocks.training.shared_initialization import load_initialization, tensor_fingerprint

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SHA256 = 'c8836a26cc2ac5e0b1c771ead4124656203d824ff86ea6aaee85fc8e66897451'


def protocol():
    spec = json.loads((ROOT / 'configs/experiments/distributional_critics.json').read_text())
    if digest(spec) != PROTOCOL_SHA256:
        raise ValueError('fixed distributional critic protocol changed')
    return spec


def overrides(arm, smoke=False):
    spec = protocol()
    if arm not in spec['arms'] or type(smoke) is not bool:
        raise ValueError('fixed critic arm and boolean smoke required')
    params = dict(spec['parameters'])
    params.pop('net_arch')
    if smoke:
        native = resolved_parameters('sac', True)
        for key in ('buffer_size', 'batch_size', 'learning_starts'):
            params[key] = native[key]
    params['policy_kwargs'] = {'net_arch': [32, 32] if smoke else [64, 64]}
    if spec['arms'][arm]['algorithm'] == 'tqc':
        params['policy_kwargs'].update(n_critics=2, n_quantiles=25)
        params['top_quantiles_to_drop_per_net'] = spec['arms'][arm]['top_quantiles_to_drop_per_net']
    return params


def identify_parameters(params, algorithm, smoke=False):
    matches = [arm for arm, value in protocol()['arms'].items()
               if value['algorithm'] == algorithm and canonical(params) == canonical(overrides(arm, smoke))]
    if len(matches) != 1:
        raise ValueError('fixed critic-study parameters and algorithm required')
    return matches[0]


def validate_payload(data):
    if (digest(data) != protocol()['dataset'] or data['manifest']['included_splits'] != ['train', 'validation']
            or len(data['names']) != 7 or any(e['split'] not in ('train', 'validation')
                or any(b['session'] > '2023-12-31' for b in e['bars']) for e in data['episodes'].values())):
        raise ValueError('fixed train/development-only critic dataset required')
    return data


def rng_record():
    return dict(torch=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                numpy=fingerprint(np.random.get_state()), python=fingerprint(random.getstate()))


def capture_global_rng():
    state = np.random.get_state()
    return dict(numpy=dict(kind=state[0], keys=state[1].tolist(), position=state[2],
                           has_gauss=state[3], cached_gaussian=state[4]),
                python=json.loads(json.dumps(random.getstate())))


def parsed_global_rng(record):
    """Validate in private generators before any global mutation."""
    value = record['numpy']
    keys = value['keys']
    if (value['kind'] != 'MT19937' or len(keys) != 624
            or any(type(v) is not int or not 0 <= v < 2**32 for v in keys)
            or type(value['position']) is not int or not 0 <= value['position'] <= 624
            or type(value['has_gauss']) is not int or value['has_gauss'] not in (0, 1)
            or not math.isfinite(value['cached_gaussian'])):
        raise ValueError('valid native NumPy sampling RNG required')
    numpy_state = (value['kind'], np.asarray(keys, dtype=np.uint32), value['position'],
                   value['has_gauss'], value['cached_gaussian'])
    python_value = record['python']
    if (len(python_value) != 3 or type(python_value[0]) is not int or python_value[0] != 3
            or len(python_value[1]) != 625
            or any(type(v) is not int for v in python_value[1])):
        raise ValueError('valid native Python sampling RNG required')
    cached = python_value[2]
    if cached is not None and (type(cached) not in (float, int) or not math.isfinite(cached)):
        raise ValueError('finite native Python cached Gaussian required')
    python_state = (python_value[0], tuple(python_value[1]), cached)
    try:
        np.random.RandomState().set_state(numpy_state)
        random.Random().setstate(python_state)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError('invalid native global RNG state') from error
    return numpy_state, python_state


def restore_global_rng(record):
    numpy_state, python_state = parsed_global_rng(record)
    np.random.set_state(numpy_state)
    random.setstate(python_state)


def model_record(model):
    if type(model).__name__ not in ('SAC', 'TQC'):
        raise ValueError('actual SAC or TQC model required')
    if type(model).__name__ == 'SAC':
        from rl_stocks.rl.tqc.geometry import validate_loaded
        validate_loaded(model, None, 'sac')
        for network in (model.critic, model.critic_target):
            if (network.n_critics != 2 or len(network.q_networks) != 2
                    or any(head[-1].out_features != 1 for head in network.q_networks)):
                raise ValueError('native two-scalar-critic SAC model required')
    if any(not bool(torch.isfinite(p).all()) for p in model.policy.parameters()):
        raise ValueError('finite initial policy tensors required')
    optimizers = {}
    for name, optimizer, parameters in (
        ('actor', model.actor.optimizer, list(model.actor.parameters())),
        ('critic', model.critic.optimizer, list(model.critic.parameters())),
        ('entropy', model.ent_coef_optimizer, [model.log_ent_coef]),
    ):
        owned = [p for group in optimizer.param_groups for p in group['params']]
        if len(owned) != len(parameters) or {id(p) for p in owned} != {id(p) for p in parameters}:
            raise ValueError('off-policy optimizer parameter ownership mismatch')
        optimizers[name] = dict(kind=type(optimizer).__module__ + '.' + type(optimizer).__name__,
            parameter_groups=[dict(lr=g['lr'], eps=g['eps'], betas=list(g['betas']),
                                   weight_decay=g['weight_decay']) for g in optimizer.param_groups],
            native_adam_semantics=dict(decoupled_weight_decay=optimizer.defaults['decoupled_weight_decay'],
                amsgrad=optimizer.defaults['amsgrad'], maximize=optimizer.defaults['maximize']),
            state_count=len(optimizer.state), state_sha256=fingerprint(optimizer.state_dict()),
            parameter_count=sum(p.numel() for p in owned))
    params = {key: getattr(model, key) for key in ('buffer_size', 'batch_size', 'learning_starts',
              'gradient_steps', 'tau', 'ent_coef', 'target_update_interval', 'n_steps', 'use_sde',
              'learning_rate', 'gamma')}
    params.update(train_freq=model.train_freq.frequency, train_freq_unit=model.train_freq.unit.value,
                  target_entropy=float(model.target_entropy),
                  net_arch=model.policy.net_arch)
    arrays = {k: v.detach().cpu().numpy() for k, v in model.policy.state_dict().items()}
    return dict(algorithm=type(model).__name__, observation_size=model.observation_space.shape[0],
                action_size=model.action_space.shape[0], seed=model.seed, num_timesteps=model.num_timesteps,
                updates=model._n_updates, parameters=params, initialization=initialization_record(model),
                policy_tensor_sha256=tensor_fingerprint(arrays), optimizers=optimizers, rng=rng_record(),
                critic_geometry=model_spec(model) if type(model).__name__ == 'TQC' else None)


def validate_initial_record(record, arm, smoke=False):
    spec = protocol()
    if arm not in spec['arms']:
        raise ValueError('fixed critic arm required')
    expected = dict(spec['parameters'])
    expected['target_entropy'] = -4.0
    expected['train_freq_unit'] = 'step'
    if smoke:
        params = resolved_parameters('sac', True)
        expected.update(net_arch=[32, 32], buffer_size=params['buffer_size'],
                        batch_size=params['batch_size'], learning_starts=params['learning_starts'])
    if (type(record['seed']) is not int or record['seed'] not in (11, 23, 37)
            or smoke and record['seed'] != 11
            or record['algorithm'].lower() != spec['arms'][arm]['algorithm']
            or canonical(record['parameters']) != canonical(expected)
            or record['observation_size'] != 21 or record['action_size'] != 4
            or type(record['num_timesteps']) is not int or type(record['updates']) is not int
            or record['num_timesteps'] != 0 or record['updates'] != 0
            or record['initialization']['alpha'] != 1.0 or record['initialization']['target_entropy'] != -4.0):
        raise ValueError('actual fresh critic-study model contract mismatch')
    for opt in record['optimizers'].values():
        if (opt['native_adam_semantics'] != dict(decoupled_weight_decay=False, amsgrad=False, maximize=False)
                or opt['state_count'] != 0 or opt['kind'] != 'torch.optim.adam.Adam'
                or opt['parameter_groups'] != [dict(lr=0.0003, eps=1e-8, betas=[0.9, 0.999], weight_decay=0)]):
            raise ValueError('fresh empty native Adam optimizers required')
    if spec['arms'][arm]['algorithm'] == 'tqc':
        geometry = record['critic_geometry']
        dropped = spec['arms'][arm]['top_quantiles_to_drop_per_net']
        expected_geometry = dict(version=1, kind='distributional_quantile_critics', n_critics=2,
                                 n_quantiles=25, total_quantiles=50, top_quantiles_to_drop_per_net=dropped,
                                 retained_target_quantiles=50 - 2 * dropped)
        if canonical(geometry) != canonical(expected_geometry):
            raise ValueError('actual fixed quantile geometry required')
    elif record['critic_geometry'] is not None:
        raise ValueError('native SAC must retain scalar critics')
    return record


def validate_initial_comparison(records, smoke=False):
    if set(records) != set(protocol()['arms']):
        raise ValueError('all three actual initial critic records required')
    for arm, record in records.items():
        validate_initial_record(record, arm, smoke)
    if len({r['seed'] for r in records.values()}) != 1:
        raise ValueError('same actual initialization seed required across arms')
    if len({r['initialization']['network_sha256']['actor'] for r in records.values()}) != 1:
        raise ValueError('shared initial actor tensors required')
    if records['tqc_untruncated']['policy_tensor_sha256'] != records['tqc_truncated']['policy_tensor_sha256']:
        raise ValueError('identical initial TQC0/TQC2 critic/target tensors required')
    if records['tqc_untruncated']['optimizers'] != records['tqc_truncated']['optimizers']:
        raise ValueError('identical initial TQC0/TQC2 optimizer state required')
    if len({digest(r['rng']) for r in records.values()}) != 1:
        raise ValueError('shared native SAC Torch/NumPy/Python sampling RNG required')
    return dict(actor_tensors_identical=True, tqc_initial_tensors_identical=True,
                tqc_initial_optimizer_states_identical=True, sampling_rng_identical=True)


def apply_initialization(model, arm, seed=11, smoke=False):
    if type(seed) is not int or model.seed != seed:
        raise ValueError('actual model seed must match initialization seed')
    validate_initial_record(model_record(model), arm, smoke)
    from rl_stocks.training.distributional_assets import initialization_entry
    entry, globals_record = initialization_entry(arm, seed, smoke)
    parsed_global_rng(globals_record)
    actual = load_initialization(model, ROOT / entry['file'], entry)
    restore_global_rng(globals_record)
    if rng_record() != entry['global_rng_sha256']:
        raise ValueError('native SAC global sampling RNG restoration failed')
    proof = dict(actual, arm=arm, seed=seed, smoke=smoke, protocol_sha256=PROTOCOL_SHA256,
                 global_rng_sha256=entry['global_rng_sha256'])
    model._research_initialization = proof
    return proof

"""Actual off-policy tensors, Adam moments and phase-sensitive normalization evidence."""

import math
import torch
from stable_baselines3 import SAC
from sb3_contrib import CrossQ

from rl_stocks.core.contracts import canonical
from rl_stocks.evaluation.soft_consistency import fingerprint
from rl_stocks.reporting.optimizer import _step
from rl_stocks.rl.crossq.geometry import model_spec, normalization_record
from rl_stocks.training.crossq_assets import initialization_spec, initialization_entry, portable_path
from rl_stocks.training.crossq_spec import parameters, expected_updates, protocol
from rl_stocks.training.distributional_critics import rng_record
from rl_stocks.training.shared_initialization import read_initialization, tensor_fingerprint


def _linear_arrays(arrays):
    weights = {k for k in arrays if k.endswith('.weight')}
    return {k: v for k, v in arrays.items() if k in weights or k.endswith('.bias') and k[:-5] + '.weight' in weights}


def _optimizer(optimizer, named):
    if type(optimizer) is not torch.optim.Adam:
        raise ValueError('native Adam required')
    owned = [p for g in optimizer.param_groups for p in g['params']]
    if len(owned) != len(named) or len({id(p) for p in owned}) != len(named) or {id(p) for p in owned} != {id(p) for _, p in named}:
        raise ValueError('exact live optimizer parameter ownership required')
    steps = {}
    for name, parameter in named:
        state = optimizer.state.get(parameter)
        if not state:
            continue
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('complete native Adam state required')
        for key in ['exp_avg', 'exp_avg_sq']:
            value = state[key]
            if value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
                raise ValueError('finite same-shape Adam moments required')
        if bool((state['exp_avg_sq'] < 0).any()):
            raise ValueError('nonnegative Adam second moment required')
        steps[name] = _step(state['step'])
    fields = ['lr', 'eps', 'betas', 'weight_decay', 'amsgrad', 'maximize', 'decoupled_weight_decay']
    return dict(kind='torch.optim.adam.Adam', groups=[{k:g[k] for k in fields} for g in optimizer.param_groups],
                parameter_count=sum(p.numel() for _, p in named), parameter_tensors=len(named),
                state_count=len(optimizer.state), per_parameter_steps=steps,
                state_sha256=fingerprint(optimizer.state_dict()))


def model_record(model):
    if type(model) not in (SAC, CrossQ):
        raise ValueError('native SAC/CrossQ model required')
    arm = 'crossq' if type(model) is CrossQ else 'sac'
    if arm == 'crossq':
        geometry = model_spec(model)
    else:
        if (model.observation_space.shape != (21,) or model.action_space.shape != (4,)
            or model.policy.net_arch != [64, 64] or len(model.critic.q_networks) != 2
            or any(net[-1].out_features != 1 for net in model.critic.q_networks)):
            raise ValueError('native full64 twin scalar SAC critics required')
        geometry = None
    if model.policy.activation_fn is not torch.nn.ReLU:
        raise ValueError('native ReLU activation required')
    arrays = {k:v.detach().cpu().numpy() for k,v in model.policy.state_dict().items()}
    if any(not bool(torch.isfinite(v).all()) for v in model.policy.state_dict().values()):
        raise ValueError('finite policy tensors/buffers required')
    if model.log_ent_coef is None or model.ent_coef_optimizer is None:
        raise ValueError('automatic entropy coefficient required')
    alpha = float(model.log_ent_coef.detach().exp())
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('finite positive entropy coefficient required')
    opts = {name:_optimizer(opt, named) for name,opt,named in [
        ('actor', model.actor.optimizer, list(model.actor.named_parameters())),
        ('critic', model.critic.optimizer, list(model.critic.named_parameters())),
        ('entropy', model.ent_coef_optimizer, [('log_ent_coef', model.log_ent_coef)])]}
    keys = ['buffer_size', 'batch_size', 'learning_starts', 'gradient_steps', 'ent_coef', 'n_steps', 'use_sde', 'learning_rate', 'gamma']
    settings = {k:getattr(model,k) for k in keys}
    settings.update(target_entropy=float(model.target_entropy), train_freq=model.train_freq.frequency,
                    train_freq_unit=model.train_freq.unit.value)
    if arm == 'crossq':
        settings['policy_delay'] = model.policy_delay
    else:
        settings.update(tau=model.tau, target_update_interval=model.target_update_interval)
    return dict(algorithm=arm, seed=model.seed, num_timesteps=model.num_timesteps, updates=model._n_updates,
                settings=settings, optimizers=opts, alpha=alpha,
                policy_tensor_sha256=tensor_fingerprint(arrays), linear_tensor_sha256=tensor_fingerprint(_linear_arrays(arrays)),
                normalization=normalization_record(model) if arm == 'crossq' else {}, geometry=geometry,
                rng=rng_record(), shared_initialization=getattr(model, '_research_initialization', None))


def _validate(record, arm, smoke, initial, phase, bound):
    if type(record['alpha']) not in (int, float) or not math.isfinite(record['alpha']) or record['alpha'] <= 0:
        raise ValueError('finite positive recorded entropy coefficient required')
    expected = parameters(arm, smoke)
    expected.pop('policy_kwargs')
    expected['target_entropy'] = -4.0
    expected['train_freq_unit'] = 'step'
    if record['algorithm'] != arm or canonical(record['settings']) != canonical(expected):
        raise ValueError('fixed actual SAC/CrossQ settings required')
    proof = initialization_spec(arm, record['seed'], smoke)
    if bound and canonical(record['shared_initialization']) != canonical(proof):
        raise ValueError('actual shared initialization provenance required')
    steps = record['num_timesteps']
    if type(steps) is not int or type(record['updates']) is not int:
        raise ValueError('integer training counters required')
    if initial:
        if steps != 0 or record['updates'] != 0 or record['alpha'] != 1:
            raise ValueError('empty initial training and entropy state required')
        counts = dict(actor=0, critic=0, entropy=0)
    else:
        cadence = protocol()['smoke_evaluation_every' if smoke else 'evaluation_every']
        if steps <= 0 or steps % cadence:
            raise ValueError('fixed checkpoint cadence required')
        counts = expected_updates(arm, steps, phase, smoke)
        if record['updates'] != counts['critic']:
            raise ValueError('actual gradient count mismatch')
    if set(record['optimizers']) != {'actor', 'critic', 'entropy'}:
        raise ValueError('all three optimizer records required')
    if not initial and phase == 'post_update' and steps != protocol()['smoke_steps' if smoke else 'steps_per_cell']:
        raise ValueError('post-update record must be the fixed final checkpoint')
    policy = parameters(arm, smoke)['policy_kwargs']['optimizer_kwargs']
    for name, opt in record['optimizers'].items():
        expected_counts = {'actor':6386 if arm == 'crossq' else 6088, 'critic':12390 if arm == 'crossq' else 11778, 'entropy':1}
        if (opt['kind'] != 'torch.optim.adam.Adam' or opt['parameter_count'] != expected_counts[name]
            or type(opt['state_count']) is not int
            or any(type(v) is not int for v in opt['per_parameter_steps'].values())):
            raise ValueError('typed native optimizer geometry required')
        group = dict(lr=0.0003, eps=1e-8, betas=[0.9,0.999] if name == 'entropy' else list(policy['betas']),
                     weight_decay=0 if name == 'entropy' else policy['weight_decay'], amsgrad=False, maximize=False, decoupled_weight_decay=False)
        if canonical(opt['groups']) != canonical([group]):
            raise ValueError('fixed effective native Adam defaults required')
        wanted = counts[name]
        if wanted == 0:
            if opt['state_count'] or opt['per_parameter_steps']:
                raise ValueError('fresh optimizer must be empty')
        elif (opt['state_count'] != opt['parameter_tensors']
              or len(opt['per_parameter_steps']) != opt['parameter_tensors']
              or set(opt['per_parameter_steps'].values()) != {wanted}):
            raise ValueError('actual per-parameter Adam steps mismatch')
    if arm == 'crossq':
        if len(record['normalization']) != 9:
            raise ValueError('nine actual normalization records required')
        for name, layer in record['normalization'].items():
            wanted = 0 if initial else counts['actor_normalization' if name.startswith('actor.') else 'critic_normalization']
            if type(layer['steps']) is not int or layer['steps'] != wanted:
                raise ValueError('actual BatchRenorm forward count mismatch')
    entry, _ = initialization_entry(arm, record['seed'], smoke)
    if initial and bound:
        if record['policy_tensor_sha256'] != entry['policy_tensor_sha256'] or record['rng'] != entry['global_rng_sha256']:
            raise ValueError('exact initial tensors and original sampling RNG required')
    elif not initial:
        arrays, _ = read_initialization(portable_path(entry['file']).read_bytes(), entry)
        if record['linear_tensor_sha256'] == tensor_fingerprint(_linear_arrays(arrays)):
            raise ValueError('learned checkpoint must change actual Linear tensors')
    return record


def validate_record(record, arm, smoke=False, *, initial=False, phase='post_update'):
    return _validate(record, arm, smoke, initial, phase, True)

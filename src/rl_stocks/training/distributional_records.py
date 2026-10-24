"""Actual SAC/TQC optimizer restoration evidence for the fixed critic study."""
import json
import math

import torch

from rl_stocks.core.contracts import canonical
from rl_stocks.evaluation.soft_consistency import fingerprint
from rl_stocks.training.distributional_assets import initialization_spec, manifest
from rl_stocks.training.distributional_critics import model_record as base_record, protocol, validate_initial_record


def _step(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError('scalar actual Adam step required')
        value = value.item()
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or int(value) != value:
        raise ValueError('nonnegative integer actual Adam step required')
    return int(value)


def optimizer_state(optimizer, parameters):
    parameters = list(parameters)
    names = {id(p): name for name, p in parameters}
    states = {}
    for parameter, state in optimizer.state.items():
        if id(parameter) not in names or set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('complete owned native Adam state required')
        for key in ('exp_avg', 'exp_avg_sq'):
            moment = state[key]
            if moment.shape != parameter.shape or moment.dtype != parameter.dtype or not bool(torch.isfinite(moment).all()):
                raise ValueError('finite matching actual Adam moments required')
        if bool((state['exp_avg_sq'] < 0).any()):
            raise ValueError('nonnegative actual Adam second moments required')
        states[names[id(parameter)]] = _step(state['step'])
    steps = list(states.values())
    return dict(parameter_count=sum(p.numel() for _, p in parameters),
                optimizer_parameter_count=len(parameters), state_count=len(states),
                per_parameter_steps=states, step_min=min(steps) if steps else None,
                step_max=max(steps) if steps else None, moments_finite=True)


def model_record(model):
    record = base_record(model)
    record['log_ent_coef_sha256'] = fingerprint({'log_ent_coef': model.log_ent_coef})
    record['shared_initialization'] = json.loads(json.dumps(getattr(model, '_research_initialization', None)))
    record['adam_state'] = dict(
        actor=optimizer_state(model.actor.optimizer, model.actor.named_parameters()),
        critic=optimizer_state(model.critic.optimizer, model.critic.named_parameters()),
        entropy=optimizer_state(model.ent_coef_optimizer, [('log_ent_coef', model.log_ent_coef)]))
    return record


def validate_record(record, arm, smoke=False, *, initial=False, selected_steps=None, seed=11):
    spec = protocol()
    proof = initialization_spec(arm, seed, smoke)
    if canonical(record.get('shared_initialization')) != canonical(proof) or record.get('seed') != seed:
        raise ValueError('actual critic initialization/seed provenance mismatch')
    probe = json.loads(json.dumps(record))
    probe.update(num_timesteps=0, updates=0)
    alpha = probe['initialization']['alpha']
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('positive finite actual learned alpha required')
    probe['initialization']['alpha'] = 1.0
    for optimizer in probe['optimizers'].values():
        optimizer['state_count'] = 0
    validate_initial_record(probe, arm, smoke)
    full_steps = spec['smoke_steps'] if smoke else spec['steps_per_cell']
    expected_steps = 0 if initial else full_steps if selected_steps is None else selected_steps
    if (type(expected_steps) is not int or not 0 <= expected_steps <= full_steps
            or not initial and expected_steps == 0
            or type(record['num_timesteps']) is not int or record['num_timesteps'] != expected_steps):
        raise ValueError('actual critic checkpoint transition count mismatch')
    warmup = record['parameters']['learning_starts']
    expected_updates = max(0, expected_steps - warmup)
    allowed = {0} if initial else {expected_updates}
    if selected_steps is not None:
        allowed.add(max(0, expected_updates - 1))
    updates = record['updates']
    if type(updates) is not int or updates not in allowed:
        raise ValueError('actual off-policy gradient count mismatch')
    group = ('smoke' if smoke else 'screen') + '-' + str(seed)
    expected = manifest()['groups'][group]['actual_initial_records'][arm]
    raw_alpha = record.get('log_ent_coef_sha256', '')
    if len(raw_alpha) != 64 or any(c not in '0123456789abcdef' for c in raw_alpha):
        raise ValueError('actual raw entropy parameter fingerprint required')
    if initial:
        if raw_alpha != fingerprint({'log_ent_coef': torch.zeros(1, dtype=torch.float32)}):
            raise ValueError('native exact initial log-alpha parameter required')
        if (record['policy_tensor_sha256'] != proof['policy_tensor_sha256']
                or record['initialization'] != expected['initialization']
                or record['rng'] != proof['global_rng_sha256']
                or record['optimizers'] != expected['optimizers']):
            raise ValueError('actual shared initial critic tensors/optimizer/RNG mismatch')
    for name, item in record['adam_state'].items():
        original = expected['optimizers'][name]
        shape_count = {'actor': 8, 'critic': 12, 'entropy': 1}[name]
        if (item['parameter_count'] != original['parameter_count']
                or item['optimizer_parameter_count'] != shape_count or item['moments_finite'] is not True
                or item['state_count'] != record['optimizers'][name]['state_count']):
            raise ValueError('actual Adam shape/finite state mismatch')
        if initial:
            if item['state_count'] != 0 or item['per_parameter_steps'] or item['step_min'] is not None or item['step_max'] is not None:
                raise ValueError('empty actual initial Adam state required')
        elif (item['state_count'] != shape_count or item['step_min'] != updates or item['step_max'] != updates
              or len(item['per_parameter_steps']) != shape_count
              or any(type(v) is not int or v != updates for v in item['per_parameter_steps'].values())):
            raise ValueError('complete matching actual learned Adam steps required')
    return record

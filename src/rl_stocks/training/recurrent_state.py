"""Frozen recurrent execution-state hypothesis and construction acceptance."""
import hashlib
import json
from pathlib import Path

from rl_stocks.core.contracts import digest

PROTOCOL_SHA256 = 'e8ea4e74426257d0e23a6a8d3068aad2836c3045ab0adc96448ea5b71a3243b8'


def protocol():
    path = Path(__file__).resolve().parents[3] / 'configs/experiments/recurrent_state.json'
    spec = json.loads(path.read_text())
    if digest(spec) != PROTOCOL_SHA256:
        raise ValueError('recurrent-state protocol changed')
    return spec


def arm_spec(arm):
    if arm not in protocol()['arms']:
        raise ValueError('fixed recurrent-state arm required')
    return dict(algorithm='ppo' if arm == 'feedforward_full' else 'recurrent_ppo',
                observation_profile='execution_masked' if arm == 'recurrent_masked' else 'execution_full')


def overrides(arm, smoke=False):
    arm_spec(arm)
    if type(smoke) is not bool:
        raise ValueError('explicit boolean smoke required')
    spec = protocol()
    params = dict(spec['parameters'])
    params.pop('net_arch')
    policy = dict(net_arch=list(spec['parameters']['net_arch']))
    if arm != 'feedforward_full':
        policy.update(spec['recurrent_parameters'])
    if smoke:
        for key in ('n_steps', 'batch_size', 'n_epochs'):
            params[key] = spec['smoke_parameters'][key]
        policy['net_arch'] = list(spec['smoke_parameters']['net_arch'])
        if arm != 'feedforward_full':
            policy['lstm_hidden_size'] = spec['smoke_parameters']['lstm_hidden_size']
    return dict(params, policy_kwargs=policy)


def model_record(model):
    state = model.policy.state_dict()
    tensor_hash = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor_hash.update(name.encode())
        tensor_hash.update(str(tuple(value.shape)).encode())
        tensor_hash.update(value.detach().cpu().numpy().tobytes())
    lstm = getattr(model.policy, 'lstm_actor', None)
    return dict(algorithm=type(model).__name__, observation_size=model.observation_space.shape[0],
                action_size=model.action_space.shape[0], num_timesteps=model.num_timesteps,
                n_steps=model.n_steps, batch_size=model.batch_size, n_epochs=model.n_epochs,
                gamma=model.gamma, gae_lambda=model.gae_lambda, ent_coef=model.ent_coef,
                learning_rate=model.learning_rate, net_arch=model.policy.net_arch,
                actor_lstm=None if lstm is None else dict(hidden_size=lstm.hidden_size, num_layers=lstm.num_layers),
                critic_lstm=getattr(model.policy, 'lstm_critic', None) is not None,
                shared_lstm=getattr(model.policy, 'shared_lstm', False),
                policy_tensor_sha256=tensor_hash.hexdigest())


def validate_record(record, arm, smoke=False, initial=False):
    expected = overrides(arm, smoke)
    algo = arm_spec(arm)['algorithm']
    fixed = dict(algorithm='PPO' if algo == 'ppo' else 'RecurrentPPO', observation_size=21, action_size=4,
                 **{key: expected[key] for key in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef', 'learning_rate')},
                 net_arch=expected['policy_kwargs']['net_arch'],
                 actor_lstm=None if algo == 'ppo' else dict(hidden_size=expected['policy_kwargs']['lstm_hidden_size'], num_layers=1),
                 critic_lstm=algo != 'ppo', shared_lstm=False)
    if any(record.get(key) != value for key, value in fixed.items()):
        raise ValueError('actual recurrent-state model configuration mismatch')
    steps = record.get('num_timesteps')
    target = 0 if initial else protocol()['smoke_steps' if smoke else 'steps_per_cell']
    if type(steps) is not int or steps != target:
        raise ValueError('actual recurrent-state model step mismatch')
    value = record.get('policy_tensor_sha256', '')
    if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('policy tensor hash required')

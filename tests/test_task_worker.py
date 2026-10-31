from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from rl_stocks.evaluation.task_gates import training_gate
from rl_stocks.training.task_conditioning import PROTOCOL_SHA256
from rl_stocks.training.task_optimizer import TaskOptimizerAudit
from rl_stocks.training.task_smoke_worker import execute_smoke_colab


def cells():
    def cell(score):
        return {'steps': 100000, 'period': 'training-window', 'phase': 'final', 'seed': 11,
                'protocol_sha256': PROTOCOL_SHA256, 'holding_path_sha256': 'a' * 64,
                'metrics': {'score': score, 'groups': {
                    'stock': {'count': 22, 'excess_return': .01, 'median_excess_return': .01},
                    'etf': {'count': 4, 'excess_return': .01}}}}
    return {'masked_shared': cell(.1), 'tagged_shared': cell(.11), 'tagged_routed': cell(.15)}


def test_training_gate_fixed_roles_and_no_promotion():
    value = cells()
    before = deepcopy(value)
    result = training_gate(value)
    assert value == before
    assert result['passed'] and result['development_allowed']
    assert not result['performance_promotion'] and not result['objective_achieved']
    value['tagged_routed']['metrics']['groups']['etf']['excess_return'] = -.03
    assert not training_gate(value)['development_allowed']
    with pytest.raises(ValueError, match='complete'):
        training_gate({'tagged_routed': value['tagged_routed']})


@pytest.mark.parametrize('field,value', [('steps', 99999), ('period', 'development'),
                                        ('seed', 23), ('phase', 'selected'),
                                        ('holding_path_sha256', 'b' * 64)])
def test_gate_rejects_incomplete_or_unmatched_inputs(field, value):
    matrix = cells()
    matrix['masked_shared'][field] = value
    with pytest.raises(ValueError):
        training_gate(matrix)


def test_worker_rejects_local_before_construction_or_outputs(tmp_path, monkeypatch):
    from rl_stocks.training import task_smoke_worker
    def forbidden(*args, **kwargs):
        pytest.fail('model construction before Colab rejection')
    monkeypatch.setattr(task_smoke_worker, 'create_task_sac', forbidden)
    with pytest.raises(RuntimeError, match='Colab'):
        execute_smoke_colab({}, {}, tmp_path / 'artifacts' / 'smoke',
                            task_role='tagged_routed')
    assert not (tmp_path / 'artifacts').exists()


def test_optimizer_observer_manual_fixture_and_hook_cleanup(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('local optimizer update attempted')
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    actor, critic = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    alpha = torch.tensor(0., requires_grad=True)
    model = SimpleNamespace(actor=actor, critic=critic, log_ent_coef=alpha,
                            ent_coef_optimizer=torch.optim.Adam([alpha]),
                            num_timesteps=0, _n_updates=0)
    actor.optimizer = torch.optim.Adam(actor.parameters())
    critic.optimizer = torch.optim.Adam(critic.parameters())
    audit = TaskOptimizerAudit(model, maximum=1)
    try:
        audit.verify(model, 0, 64)
        for role, optimizer in audit.optimizers.items():
            for parameter in audit.named[role].values():
                parameter.grad = torch.ones_like(parameter)
            audit.before(role, optimizer)
            # Manually assigned tensor fixtures, never a learner update.
            for parameter in audit.named[role].values():
                optimizer.state[parameter] = {'step': torch.tensor(1.),
                                             'exp_avg': torch.ones_like(parameter),
                                             'exp_avg_sq': torch.ones_like(parameter)}
            audit.after(role, optimizer)
        model.num_timesteps, model._n_updates = 65, 1
        audit.verify(model, 65, 64)
        with pytest.raises(ValueError, match='bounded'):
            audit.before('actor', actor.optimizer)
        first = next(iter(actor.parameters()))
        actor.optimizer.state[first]['exp_avg_sq'].fill_(-1)
        with pytest.raises(ValueError, match='moments'):
            audit.verify(model, 65, 64)
    finally:
        audit.close()
    assert not audit.handles
    assert not actor.optimizer._optimizer_step_pre_hooks


def test_worker_closes_constructed_environment_on_failure(tmp_path, monkeypatch):
    from test_task_environment import payload
    from rl_stocks.core.contracts import digest
    from rl_stocks.training import task_smoke_worker as worker
    from rl_stocks.training.task_conditioning import protocol
    data, evidence = payload()
    spec = deepcopy(protocol())
    spec['dataset'], spec['normalizer_sha256'] = digest(data), digest(data['normalizer'])
    monkeypatch.setattr(worker, 'require_colab_cpu', lambda: None)
    monkeypatch.setattr(worker, 'protocol', lambda: spec)
    monkeypatch.setattr(worker, 'repository_state', lambda _: {'revision': 'a' * 40, 'dirty': ''})
    closed = []
    environment = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(worker, 'build_task_environment', lambda *a, **k: environment)
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic constructor failure before learn')
    monkeypatch.setattr(worker, 'create_task_sac', fail)
    destination = tmp_path / 'artifacts' / 'smoke'
    with pytest.raises(RuntimeError, match='synthetic'):
        worker.execute_smoke_colab(data, evidence, destination, task_role='tagged_shared')
    assert closed == [True]
    assert (destination / 'failure.json').is_file()
    assert not (destination / 'run.json').exists()


def test_task_restoration_checks_actual_saved_models_without_training(tmp_path, monkeypatch):
    import numpy as np
    from stable_baselines3 import SAC
    from test_task_bundle import bundle
    from rl_stocks.training.task_smoke_worker import verify_restoration
    def forbidden(*args, **kwargs):
        pytest.fail('local optimization attempted')
    monkeypatch.setattr(SAC, 'learn', forbidden)
    monkeypatch.setattr(SAC, 'train', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    model, path, _ = bundle(tmp_path)
    x = np.zeros((1, 22), dtype=np.float32)
    x[:, -1] = 1
    model.replay_buffer.add(x, x, np.zeros((1, 4)), np.zeros(1), np.zeros(1), [{}])
    replay = tmp_path / 'replay.pkl'
    model.save_replay_buffer(replay)
    restored = SAC.load(path / 'model.zip')
    restored.load_replay_buffer(replay)
    verify_restoration(model, restored)
    restored.replay_buffer.observations[0, 0, 0] = 1
    with pytest.raises(ValueError, match='replay restoration'):
        verify_restoration(model, restored)
    restored.load_replay_buffer(replay)
    with torch.no_grad():
        restored.log_ent_coef.add_(1)
    with pytest.raises(ValueError, match='optimizer restoration'):
        verify_restoration(model, restored)

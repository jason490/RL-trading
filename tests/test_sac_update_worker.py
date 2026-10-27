"""Preparation/manual serialization only. Local native updates are forbidden."""
import json
import random

import numpy as np
import pytest
from stable_baselines3 import SAC
import torch

from test_sac_update_input import fixture
from rl_stocks.core.contracts import canonical, digest
from rl_stocks.training import sac_update_worker as worker
from rl_stocks.training.distributional_critics import rng_record
from rl_stocks.training.shared_initialization import tensor_fingerprint


@pytest.fixture(autouse=True)
def forbid_local_optimization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('local optimizer update or learning attempted')
    monkeypatch.setattr(SAC, 'learn', forbidden)
    monkeypatch.setattr(SAC, 'train', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)


def prepared(monkeypatch, arm='original_defaults'):
    payload = fixture()
    monkeypatch.setattr(worker, 'INPUT_ID', digest(payload))
    model, initial = worker.prepare(payload, arm)
    return model, initial, payload


def test_copied_starts_rng_replay_and_native_container_representations(monkeypatch):
    records, actions, types = {}, {}, {}
    before = canonical(rng_record())
    payload = fixture()
    immutable_input = canonical(payload)
    monkeypatch.setattr(worker, 'INPUT_ID', digest(payload))
    with worker.preserved_rng():
        for arm in ('original_defaults', 'explicit_serialized_defaults'):
            model, record = worker.prepare(payload, arm)
            try:
                records[arm] = record
                actions[arm] = worker.stochastic_action(model, payload['postupdate_observation'])
                types[arm] = type(model.actor.optimizer.param_groups[0]['betas'])
                assert model.replay_buffer.pos == 1001 and model._n_updates == 0
                assert model.num_timesteps == 1001 and model._current_progress_remaining == 0.97998
                assert not model.actor.optimizer.state
            finally:
                model.env.close()
    assert canonical(rng_record()) == before
    assert canonical(payload) == immutable_input
    a, b = records.values()
    assert a['policy_tensor_sha256'] == b['policy_tensor_sha256'] and a['rng'] == b['rng']
    np.testing.assert_array_equal(*actions.values())
    assert types == dict(original_defaults=tuple, explicit_serialized_defaults=list)


def test_sample_and_manual_observer_budget_guards(monkeypatch):
    with worker.preserved_rng():
        model, _, _ = prepared(monkeypatch)
        audit = worker.SingleUpdateAudit(model)
        try:
            assert len(model.actor.optimizer._optimizer_step_pre_hooks) == 1
            audit._sample(256)  # Native replay sampling is permitted; no optimizer call.
            before = canonical(rng_record())
            with pytest.raises(ValueError, match='exactly one native replay batch'):
                audit._sample(256)
            assert canonical(rng_record()) == before
            # Manual callbacks only, not evidence of actual optimizer steps.
            for name in ('entropy', 'critic', 'actor'):
                audit._before_step(name)
                audit._after_step(name)
            with pytest.raises(ValueError, match='exactly one step'):
                audit._before_step('actor')
            assert audit.record()['optimizer_steps'] == dict(actor=1, critic=1, entropy=1)
        finally:
            audit.close()
            audit.close()
            model.env.close()
        assert 'sample' not in model.replay_buffer.__dict__
        assert not model.actor.optimizer._optimizer_step_pre_hooks
        assert not model.actor.optimizer._optimizer_step_post_hooks
        assert model._n_updates == 0 and not model.actor.optimizer.state


def test_manual_tensor_moment_npz_and_native_save_load(tmp_path, monkeypatch):
    with worker.preserved_rng():
        model, _, _ = prepared(monkeypatch)
        try:
            for optimizer in (model.actor.optimizer, model.critic.optimizer, model.ent_coef_optimizer):
                for group in optimizer.param_groups:
                    for parameter in group['params']:
                        optimizer.state[parameter] = dict(step=torch.tensor(1.),
                            exp_avg=torch.full_like(parameter, 0.01), exp_avg_sq=torch.full_like(parameter, 0.02))
            arrays = worker.state_arrays(model)
            np.savez_compressed(tmp_path / 'state.npz', **arrays)
            with np.load(tmp_path / 'state.npz', allow_pickle=False) as archive:
                assert tensor_fingerprint({k:archive[k] for k in archive.files}) == tensor_fingerprint(arrays)
            model.save(tmp_path / 'fixture.zip')
            restored = SAC.load(tmp_path / 'fixture.zip', device='cpu')
            assert tensor_fingerprint(worker.state_arrays(restored)) == tensor_fingerprint(arrays)
        finally:
            model.env.close()


def test_worker_rejects_local_execution_before_construction_or_output(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('local execute reached preparation')
    monkeypatch.setattr(worker, 'prepare', forbidden)
    with pytest.raises(RuntimeError, match='local updates forbidden'):
        worker.execute_colab(fixture(), tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_rng_restores_on_failure():
    before = canonical(rng_record())
    with pytest.raises(RuntimeError):
        with worker.preserved_rng():
            random.random()
            np.random.normal()
            torch.rand(5)
            raise RuntimeError('fixture')
    assert canonical(rng_record()) == before


def test_invalid_bound_input_rejected_before_model_constructor(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('bad input reached model construction')
    monkeypatch.setattr(worker, 'SAC', forbidden)
    with pytest.raises(ValueError, match='exact bound numerical input'):
        worker.prepare(fixture(), 'original_defaults')
    json.dumps(fixture(), allow_nan=False)


def test_failed_step_attempt_is_not_reported_as_completed(monkeypatch):
    with worker.preserved_rng():
        model, _, _ = prepared(monkeypatch)
        audit = worker.SingleUpdateAudit(model)
        try:
            audit._before_step('actor')  # Manual pre-hook fixture, no optimizer call.
            assert audit.record()['optimizer_step_attempts']['actor'] == 1
            assert audit.record()['optimizer_steps']['actor'] == 0
            with pytest.raises(ValueError, match='exactly one step'):
                audit._before_step('actor')
        finally:
            audit.close()
            model.env.close()

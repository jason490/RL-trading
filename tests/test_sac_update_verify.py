"""Manual Adam fixtures and forward-only RNG replay; never fit a model locally."""
import json

import numpy as np
import pytest
from stable_baselines3 import SAC
import torch

from test_sac_update_input import fixture
from rl_stocks.core.contracts import canonical, digest
from rl_stocks.evaluation.soft_consistency import fingerprint
from rl_stocks.training import sac_update_worker as worker
from rl_stocks.training.distributional_critics import rng_record
from rl_stocks.training.crossq_state import model_record
from rl_stocks.training.sac_update_verify import verify_condition, verify_directory
from rl_stocks.training.shared_initialization import tensor_fingerprint


@pytest.fixture(autouse=True)
def no_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('local optimizer step or training attempted')
    monkeypatch.setattr(SAC, 'learn', forbidden)
    monkeypatch.setattr(SAC, 'train', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)


def saved_fixture(tmp_path, monkeypatch, arm='original_defaults'):
    payload = fixture()
    monkeypatch.setattr(worker, 'INPUT_ID', digest(payload))
    with worker.preserved_rng():
        model, initial = worker.prepare(payload, arm)
        try:
            torch_states = dict(initial=torch.get_rng_state().numpy().copy())
            before_action = worker.stochastic_action(model, payload['arrays']['observations']['values'][1000][0])
            before_rng = rng_record()
            torch_states['before_update'] = torch.get_rng_state().numpy().copy()
            audit = worker.SingleUpdateAudit(model)
            try:
                sample = audit._sample(256)
                # Native forward calls only, not backward/optimizer/training.
                with torch.no_grad():
                    model.actor.action_log_prob(sample.observations)
                    model.actor.action_log_prob(sample.next_observations)
                # Manual serialization state; never an actual optimizer update.
                for name, optimizer in (('entropy', model.ent_coef_optimizer), ('critic', model.critic.optimizer),
                                        ('actor', model.actor.optimizer)):
                    audit._before_step(name)
                    for group in optimizer.param_groups:
                        for parameter in group['params']:
                            optimizer.state[parameter] = dict(step=torch.tensor(1.),
                                exp_avg=torch.full_like(parameter, 0.01), exp_avg_sq=torch.full_like(parameter, 0.02))
                    audit._after_step(name)
                with torch.no_grad():
                    model.actor.mu.weight.add_(0.001)
                    model.log_ent_coef.fill_(-0.0003)
                model._n_updates = 1
                after_update = rng_record()
                torch_states['after_update'] = torch.get_rng_state().numpy().copy()
                final = model_record(model)
                action = worker.stochastic_action(model, payload['postupdate_observation'])
                torch_states['after_action'] = torch.get_rng_state().numpy().copy()
                arrays = worker.state_arrays(model)
                record = dict(initial=initial, final=final, audit=audit.record(), before_action=before_action.tolist(),
                    before_action_sha256=fingerprint(before_action), before_rng=before_rng,
                    after_update_rng=after_update, after_action=action.tolist(), after_action_sha256=fingerprint(action),
                    after_rng=rng_record(), state_sha256=tensor_fingerprint(arrays))
                tmp_path.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(tmp_path / 'torch-rng.npz', **torch_states)
                np.savez_compressed(tmp_path / 'sampled-batch.npz', **audit.batch)
                np.savez_compressed(tmp_path / 'numerical-state.npz', **arrays)
            finally:
                audit.close()
            model.save(tmp_path / 'post-update.zip')
            (tmp_path / 'record.json').write_bytes(canonical(record))
        finally:
            model.env.close()
    return payload


@pytest.mark.parametrize('arm', ['original_defaults', 'explicit_serialized_defaults'])
def test_actual_native_deserialization_and_forward_rng_without_fitting(tmp_path, monkeypatch, arm):
    payload = saved_fixture(tmp_path, monkeypatch, arm)
    before = canonical(rng_record())
    result = verify_condition(tmp_path, payload, arm)
    assert canonical(rng_record()) == before
    assert result['actual_saved_model_and_adam_restored']
    assert result['sampled_batch_reconstructed'] and result['forward_only_rng_replay']
    assert result['remote_preupdate_action_recomputed_exact'] and result['remote_postupdate_action_recomputed_exact']
    assert result['optimizer_steps'] == 0 and result['training_performed'] is False


@pytest.mark.parametrize('change', ['batch', 'tensor', 'rng', 'action', 'counts', 'moment'])
def test_mutated_diagnostic_artifacts_rejected_and_rng_restored(tmp_path, monkeypatch, change):
    payload = saved_fixture(tmp_path, monkeypatch)
    if change in ('batch', 'tensor', 'rng'):
        file = {'batch':'sampled-batch.npz', 'tensor':'numerical-state.npz', 'rng':'torch-rng.npz'}[change]
        with np.load(tmp_path / file, allow_pickle=False) as archive:
            arrays = {k:archive[k].copy() for k in archive.files}
        key = next(iter(arrays))
        arrays[key].flat[0] += 1
        np.savez_compressed(tmp_path / file, **arrays)
    elif change in ('action', 'counts'):
        path = tmp_path / 'record.json'
        record = json.loads(path.read_text())
        if change == 'action':
            record['after_action'][0] += 0.01
        else:
            record['audit']['optimizer_steps']['actor'] = 2
        path.write_text(json.dumps(record))
    else:
        model = SAC.load(tmp_path / 'post-update.zip', device='cpu')
        next(iter(model.actor.optimizer.state.values()))['exp_avg'].add_(0.1)
        model.save(tmp_path / 'post-update.zip')
    before = canonical(rng_record())
    with pytest.raises((ValueError, RuntimeError)):
        verify_condition(tmp_path, payload, 'original_defaults')
    assert canonical(rng_record()) == before


def directory_fixture(tmp_path, monkeypatch, different=False):
    records, arrays = {}, {}
    for arm in worker.protocol()['arms']:
        payload = saved_fixture(tmp_path / arm, monkeypatch, arm)
        path = tmp_path / arm
        if different and arm == 'explicit_serialized_defaults':
            model = SAC.load(path / 'post-update.zip', device='cpu')
            with torch.no_grad():
                model.critic.q_networks[0][-1].bias.add_(0.001)
            model.save(path / 'post-update.zip')
            record = json.loads((path / 'record.json').read_text())
            record['final'] = model_record(model)
            state = worker.state_arrays(model)
            record['state_sha256'] = tensor_fingerprint(state)
            np.savez_compressed(path / 'numerical-state.npz', **state)
            (path / 'record.json').write_bytes(canonical(record))
        records[arm] = json.loads((path / 'record.json').read_text())
        with np.load(path / 'numerical-state.npz', allow_pickle=False) as archive:
            arrays[arm] = {k:archive[k].copy() for k in archive.files}
    a,b = worker.protocol()['arms']
    left,right = records[a],records[b]
    comparison = dict(initial_policy_exact=True, preupdate_action_exact=True, preupdate_rng_exact=True, sampled_batch_exact=True,
        postupdate_different_fields=[k for k in sorted(arrays[a]) if fingerprint(arrays[a][k]) != fingerprint(arrays[b][k])],
        postupdate_tensor_state_exact=left['state_sha256'] == right['state_sha256'],
        postupdate_rng_exact=left['after_rng'] == right['after_rng'],
        postupdate_action_exact=left['after_action_sha256'] == right['after_action_sha256'])
    summary = dict(protocol_sha256=worker.PROTOCOL_SHA256, input_identity=worker.INPUT_ID, status='complete',
        execution_backend=worker.execution_backend('cpu'), records=records, comparison=comparison,
        total_model_gradient_updates=2, total_adam_steps=6, environment_transitions=0,
        performance_evaluated=False, crossq_stop_cleared=False, root_cause_established=False)
    (tmp_path / 'summary.json').write_bytes(canonical(summary))
    return payload


@pytest.mark.parametrize('different', [False, True])
def test_recomputed_comparison_accepts_either_numerical_outcome_without_promotion(tmp_path, monkeypatch, different):
    payload = directory_fixture(tmp_path, monkeypatch, different)
    result = verify_directory(tmp_path, payload)
    assert result['saved_artifacts_verified']
    assert result['comparison']['postupdate_tensor_state_exact'] is (not different)
    assert result['root_cause_established'] is False and result['crossq_stop_cleared'] is False


@pytest.mark.parametrize('field,value', [('crossq_stop_cleared', True), ('root_cause_established', True),
                                        ('environment_transitions', False), ('total_adam_steps', 7)])
def test_summary_cannot_authorize_more_work_or_reinterpret_protocol(tmp_path, monkeypatch, field, value):
    payload = directory_fixture(tmp_path, monkeypatch)
    path = tmp_path / 'summary.json'
    summary = json.loads(path.read_text())
    summary[field] = value
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        verify_directory(tmp_path, payload)

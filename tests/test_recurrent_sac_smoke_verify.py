"""Manual untrained/Adam/target fixtures; no local gradient or optimizer update."""
from copy import deepcopy
import json

import numpy as np
import pytest
from stable_baselines3 import SAC
import torch

from test_sac_update_input import fixture
from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.execution_provenance import execution_backend
from rl_stocks.training import recurrent_sac_smoke as smoke, sac_update_worker as old_worker
from rl_stocks.training import recurrent_sac_smoke_verify as verifier
from rl_stocks.training import recurrent_sac_smoke_worker as worker
from rl_stocks.training.crossq_state import model_record
from rl_stocks.training.recurrent_sac_checkpoint import save_checkpoint


@pytest.fixture(autouse=True)
def no_fitting(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail('local optimization/differentiation forbidden')
    monkeypatch.setattr(SAC, 'learn', forbidden)
    monkeypatch.setattr(SAC, 'train', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden)


def manual_state(optimizer):
    for group in optimizer.param_groups:
        for param in group['params']:
            optimizer.state[param] = dict(step=torch.tensor(4.), exp_avg=torch.full_like(param, .01),
                                           exp_avg_sq=torch.full_like(param, .02))


def directory_fixture(root, monkeypatch):
    payload = fixture()
    spec = smoke.protocol()
    spec['input_identity'] = digest(payload)
    for module in (smoke, worker, verifier):
        monkeypatch.setattr(module, 'protocol', lambda: deepcopy(spec))
    monkeypatch.setattr(old_worker, 'INPUT_ID', digest(payload))
    root.mkdir(parents=True)
    with old_worker.preserved_rng():
        prepared = smoke.prepare_smoke(payload)
        try:
            learner, native = prepared.learner, prepared.control
            lineage = dict(source='recurrent-sac-smoke-'+smoke.PROTOCOL_SHA256,
                           dataset=digest(payload), observation_schema='execution-full-v1-normalized21')
            save_checkpoint(learner, root/'recurrent-initial', lineage=lineage)
            native.save(root/'native-initial.zip')
            for update in range(1, 5):
                old = {name: p.detach().clone() for name, p in learner.networks.targets.named_parameters()}
                arrays = {}
                with torch.no_grad():
                    for p in learner.networks.critics.parameters():
                        p.add_(.001)  # fixed manual serialization fixture, independent of data/loss
                    for (name, source), target in zip(learner.networks.critics.named_parameters(),
                                                       learner.networks.targets.parameters(), strict=True):
                        target.copy_(old[name].mul(1-learner.tau).add(source, alpha=learner.tau))
                        for prefix, tensor in [('old', old[name]), ('source', source), ('actual', target)]:
                            arrays[prefix+'/'+name] = tensor.detach().numpy().copy()
                np.savez_compressed(root/f'polyak-{update}.npz', **arrays)
            with torch.no_grad():
                learner.networks.actor.mean.bias.add_(.001)
                learner.log_alpha.fill_(-.001)
                native.actor.mu.weight.add_(.001)
                native.log_ent_coef.fill_(-.001)
            learner.updates = learner.attempts = 4
            learner.completed_steps = dict.fromkeys(learner.optimizers, 4)
            for opt in learner.optimizers.values():
                manual_state(opt)
            for opt in (native.actor.optimizer, native.critic.optimizer, native.ent_coef_optimizer):
                manual_state(opt)
            native._n_updates = 4
            save_checkpoint(learner, root/'recurrent-final', lineage=lineage)
            native.save(root/'native-final.zip')
            records = [dict(update=i, polyak_exact=True, order=['entropy', 'critic', 'actor', 'targets'],
                            completed_steps=dict.fromkeys(('entropy', 'critic', 'actor'), i),
                            actor_loss=.1, critic_loss=.2, entropy_loss=.3, alpha_before=1., alpha_after=.99)
                       for i in range(1, 5)]
            events = [dict(role=role, parameter_count=1, finite_gradients=True,
                           nonzero_memory_gradients={'entropy': [], 'critic': [True, True], 'actor': [True]}[role])
                      for _ in range(4) for role in ('entropy', 'critic', 'actor')]
            summaries = dict(protocol_sha256=smoke.PROTOCOL_SHA256, input_identity=digest(payload),
                             model_updates=8, adam_steps=24, recurrent_updates=4, native_updates=4,
                             prediction_count=9, performance_evaluation=False, environment_transitions=0,
                             crossq_stop_cleared=False, backend=execution_backend('cpu'))
            summaries['backend']['torch_intraop_threads'] = 1
            files = {'input.json': payload, 'summary.json': summaries,
                     'recurrent-audit.json': dict(events=events, records=records,
                                                  steps=dict.fromkeys(('entropy', 'critic', 'actor'), 4)),
                     'recurrent-progress.json': records,
                     'recurrent-predictions.json': worker.prediction_parity(learner, learner, prepared.arrays),
                     'native-audit.json': dict(events=events, final_state=model_record(native),
                                               steps=dict.fromkeys(('entropy', 'critic', 'actor'), 4)),
                     'loss-rows.json': [r.tolist() for r in prepared.loss_rows]}
            for name, value in files.items():
                (root/name).write_bytes(canonical(value))
            return payload
        finally:
            prepared.control.get_env().close()


def test_physical_manual_snapshots_target_chain_and_row_identity_without_fitting(tmp_path, monkeypatch):
    directory_fixture(tmp_path/'smoke', monkeypatch)
    result = verifier.verify_directory(tmp_path/'smoke')
    assert result['saved_artifacts_verified'] and result['polyak_chain_verified']
    assert result['matched_loss_rows'] and result['local_frozen_prediction_parity']
    assert result['model_gradient_updates'] == 8  # manual fixture metadata, never actual learning


@pytest.mark.parametrize('field', ['rows', 'polyak', 'events', 'summary', 'counter'])
def test_corrupted_physical_evidence_rejects(tmp_path, monkeypatch, field):
    root = tmp_path/'smoke'
    directory_fixture(root, monkeypatch)
    if field == 'polyak':
        path = root/'polyak-2.npz'
        with np.load(path, allow_pickle=False) as values:
            arrays = {k: values[k].copy() for k in values.files}
        arrays[next(k for k in arrays if k.startswith('actual/'))].flat[0] += .1
        np.savez_compressed(path, **arrays)
    else:
        name = {'rows': 'loss-rows.json', 'events': 'recurrent-audit.json',
                'summary': 'summary.json', 'counter': 'recurrent-final/manifest.json'}[field]
        value = json.loads((root/name).read_text())
        if field == 'rows':
            value[0][0] += 1
        elif field == 'events':
            value['events'][1]['nonzero_memory_gradients'] = [False, True]
        elif field == 'summary':
            value['performance_evaluation'] = True
        else:
            value['updates'] = 3
        (root/name).write_bytes(canonical(value))
    with pytest.raises(ValueError):
        verifier.verify_directory(root)

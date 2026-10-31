from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl_stocks.core.contracts import digest
from rl_stocks.evaluation import offpolicy_fit as audit


def test_fixed_scope_and_original_train_labels_with_unmasked_carried_state():
    from test_core import bar
    from test_training_fit import data
    spec = audit.protocol()
    assert spec['arms'] == ['gru_sac', 'native_sac'] and spec['phases'] == ['selected', 'final']
    source = data()
    episode = source['episodes']['train']
    episode['bars'] = [asdict(bar(d)) for d in
                       ['2017-12-28', '2017-12-29', '2018-01-02', '2020-12-02', '2020-12-03']]
    second = deepcopy(episode)
    second['symbol'] = 'AAPL'
    source['episodes']['second'] = second
    source['manifest']['evaluation_symbols'].append('AAPL')
    source.update(names=['fixture'], normalizer=dict(names=['fixture'], means=[0.], scales=[1.]))
    class Model:
        policy = SimpleNamespace(lstm_actor=True)
        def __init__(self):
            self.starts = []
        def predict(self, obs, *, state, episode_start, deterministic):
            assert deterministic and (state is None) == bool(episode_start[0])
            if not episode_start[0]:
                assert np.any(obs[-9:] != 0)  # Existing protection must not be masked.
            self.starts.append(bool(episode_start[0]))
            return np.array([.25, 1., 1., 1.]), (np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))
    before = deepcopy(source)
    model = Model()
    result = audit.evaluate_fixed_window(model, source)
    assert model.starts == [True, False, True, False] and source == before
    assert all(r['split'] == 'train' for r in result.values())
    assert result['train']['rows'][0]['policy_action'][0] == .25
    assert result['train']['rows'][0]['raw_action'][0] == 1.
    source['episodes']['second']['bars'][-1]['session'] = '2024-01-02'
    with pytest.raises(ValueError):
        audit.evaluate_fixed_window(model, source)


def test_no_fitting_guard_blocks_real_entry_points_and_restores_them():
    from stable_baselines3 import SAC
    from rl_stocks.data.generation.datasets import Normalizer
    from rl_stocks.training.recurrent_sac_learner import RecurrentSACLearner
    original = torch.optim.Adam.step
    with audit.inference_only():
        for call in [lambda: SAC.learn(None), lambda: SAC.train(None),
                     lambda: RecurrentSACLearner.update(None, None),
                     lambda: torch.optim.Adam.step(None),
                     lambda: torch.tensor(1., requires_grad=True).backward(),
                     lambda: Normalizer.fit(None, None)]:
            with pytest.raises(RuntimeError, match='forbidden'):
                call()
    assert torch.optim.Adam.step is original


def test_fingerprint_detects_moments_alpha_and_counter_mutation():
    value = dict(weights=torch.tensor([1.]), alpha=torch.tensor(0.),
                 optimizer={0:dict(exp_avg=torch.tensor([.1]), step=torch.tensor(9.))}, updates=9)
    baseline = audit.fingerprint(value)
    for change in [lambda x:x['weights'].add_(1), lambda x:x['alpha'].add_(1),
                   lambda x:x['optimizer'][0]['exp_avg'].add_(1), lambda x:x.update(updates=10)]:
        changed = deepcopy(value)
        change(changed)
        assert audit.fingerprint(changed) != baseline
    assert audit.fingerprint(value) == baseline


def test_cell_tampering_rejects_before_model_loading(tmp_path, monkeypatch):
    from rl_stocks.evaluation.recurrent_comparison_acceptance import receipt_hash
    folder = tmp_path/'cell'
    folder.mkdir()
    (folder/'weights').write_bytes(b'fixture')
    receipt = dict(dataset='fixture', stage='screen', gate=dict(passed=False), retired_test_access=False,
                   source_cells={'native_sac-11':dict(folder=str(folder))},
                   cells={'11':{'native_sac':dict(saved_artifacts_verified=True,
                           artifact_inventory_sha256=digest(audit.inventory(folder)))}})
    receipt['receipt_sha256'] = receipt_hash(receipt)
    path = tmp_path/'acceptance.json'
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    monkeypatch.setattr(audit, 'protocol', lambda: dict(acceptance='acceptance.json',
                        acceptance_receipt_sha256=receipt['receipt_sha256'], dataset='fixture', arms=['native_sac']))
    assert set(audit.accepted_cells()[1]) == {'native_sac'}
    (folder/'weights').write_bytes(b'changed')
    with pytest.raises(ValueError, match='inventory'):
        audit.accepted_cells()

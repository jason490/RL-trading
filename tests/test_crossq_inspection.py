import hashlib
import pickle
import random

import numpy as np
import pytest
from sb3_contrib import CrossQ
from stable_baselines3 import SAC
import torch

from rl_stocks.training import crossq_inspection as inspection


def test_actual_untrained_geometry_buffers_rng_and_serialization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('inspection attempted optimization')

    for cls in [CrossQ, SAC]:
        monkeypatch.setattr(cls, 'learn', forbidden)
        monkeypatch.setattr(cls, 'train', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    numpy_before = hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest()
    python_before, torch_before = random.getstate(), torch.get_rng_state().clone()
    result = inspection.audit()
    assert numpy_before == hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest()
    assert python_before == random.getstate() and torch.equal(torch_before, torch.get_rng_state())
    sac, crossq = result['records']['sac'], result['records']['crossq']
    assert (sac['actor_parameters'], sac['critic_parameters']) == (6088, 11778)
    assert (crossq['actor_parameters'], crossq['critic_parameters']) == (6386, 12390)
    assert sac['target_critic_parameters'] == 11778 and sac['target_critic_present']
    assert not crossq['target_critic_present'] and crossq['target_critic_parameters'] == 0
    assert not sac['normalization'] and len(crossq['normalization']) == 9
    assert all(v['steps'] == 0 and v['warmup_steps'] == 100000 for v in crossq['normalization'].values())
    assert sac['optimizers']['actor']['betas'] == [0.9, 0.999]
    assert crossq['optimizers']['actor']['betas'] == [0.5, 0.999]
    assert result['crossq_defaults']['policy_delay'] == 3
    assert result['optimizer_steps'] == result['transitions'] == result['colab_units'] == 0
    for row in result['records'].values():
        assert row['exact_untrained_prediction_parity'] and row['normalization_and_weights_unchanged']
        assert row['empty_replay_restored']
    assert not result['actual_learned_acceptance'] and not result['performance_protocol_registered']


def test_version_drift_rejects_before_model_construction(monkeypatch):
    monkeypatch.setattr(inspection.importlib.metadata, 'version', lambda name: 'unexpected')
    monkeypatch.setattr(inspection, 'SpacesOnly', lambda spec: pytest.fail('constructed before version check'))
    with pytest.raises(ValueError, match='review installed'):
        inspection.audit()

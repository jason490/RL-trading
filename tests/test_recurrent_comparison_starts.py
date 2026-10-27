"""Actual untrained constructors and serialization; optimization forbidden."""
import json

import pytest
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
import torch

from rl_stocks.training.distributional_critics import rng_record
from rl_stocks.training.recurrent_comparison_starts import build,accept,construct
from rl_stocks.training.recurrent_sac_learner import RecurrentSACLearner


def test_all_nine_actual_untrained_starts_restore_without_mutating_streams(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('local fitting attempted')
    for cls in (SAC,RecurrentPPO):
        monkeypatch.setattr(cls,'learn',forbidden)
        monkeypatch.setattr(cls,'train',forbidden)
    monkeypatch.setattr(RecurrentSACLearner,'update',forbidden)
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
    before=rng_record()
    destination=tmp_path/'starts'
    manifest=build(destination)
    assert rng_record()==before and len(manifest['entries'])==9
    assert accept(destination)['cells']==9
    assert rng_record()==before
    with pytest.raises(FileExistsError):
        build(destination)
    path=destination/'native_sac-11.npz'
    path.write_bytes(path.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='checksum'):
        accept(destination)
    assert rng_record()==before


def test_unknown_roles_and_seeds_reject_before_rng_changes():
    before=rng_record()
    for arm,seed in [('unknown',11),('gru_sac',True),('native_sac',7)]:
        with pytest.raises(ValueError):
            construct(arm,seed)
    assert rng_record()==before


def test_incomplete_matrix_rejects(tmp_path):
    destination=tmp_path/'starts'
    build(destination)
    path=destination/'manifest.json'
    value=json.loads(path.read_text())
    del value['entries']['gru_sac-37']
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='manifest'):
        accept(destination)

"""Untrained save/count contracts and guards; no actual local training."""
from dataclasses import replace
import json

import pytest
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
import torch

from rl_stocks.training import recurrent_comparison_worker as worker
from rl_stocks.training.distributional_critics import rng_record
from rl_stocks.training.recurrent_comparison import candidate_config
from rl_stocks.training.recurrent_comparison_starts import construct
from rl_stocks.training.recurrent_loop import validate_initial_learner
from rl_stocks.training.recurrent_sac_learner import RecurrentSACLearner
from rl_stocks.training.sac_update_worker import preserved_rng


@pytest.fixture(autouse=True)
def no_fitting(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('local training or optimizer step attempted')
    for cls in (SAC,RecurrentPPO):
        monkeypatch.setattr(cls,'learn',forbidden)
        monkeypatch.setattr(cls,'train',forbidden)
    monkeypatch.setattr(RecurrentSACLearner,'update',forbidden)
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)


def test_full_worker_rejects_locally_before_any_io_or_construction(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('full cell reached model construction locally')
    monkeypatch.setattr(worker,'load_start',forbidden)
    before=rng_record()
    output=tmp_path/'artifacts'/'full'
    with pytest.raises(RuntimeError,match='Colab CPU'):
        worker.execute_colab({},tmp_path/'starts',output,arm='gru_sac',seed=11)
    assert not output.exists() and rng_record()==before


def test_native_counts_distinguish_actual_adam_steps_and_ppo_epochs():
    assert worker.expected_counts('native_sac',10000)==(9000,dict(actor=9000,critic=9000,entropy=9000))
    assert worker.expected_counts('native_sac',100000)[0]==99000
    assert worker.expected_counts('recurrent_ppo_reference',10000)==(100,dict(policy=2000))
    assert worker.expected_counts('recurrent_ppo_reference',100000)==(1000,dict(policy=20000))
    for arm,step in [('native_sac',True),('native_sac',100001),('recurrent_ppo_reference',999),('td3',1000)]:
        with pytest.raises(ValueError):
            worker.expected_counts(arm,step)


@pytest.mark.parametrize('arm',['native_sac','recurrent_ppo_reference'])
def test_actual_untrained_native_snapshot_roundtrip_and_counter_rejection(tmp_path,arm):
    with preserved_rng():
        model=construct(arm,11)
        audit=worker.NativeAudit(model,arm)
        try:
            folder=worker.native_snapshot(model,arm,11,0,audit,tmp_path,dict(source='untrained-fixture'))
            manifest=json.loads((tmp_path/folder/'manifest.json').read_text())
            assert manifest['transitions']==0 and manifest['optimizer_steps']==dict.fromkeys(audit.counts,0)
            restored=type(model).load(tmp_path/folder/'model.zip',device='cpu')
            assert restored.num_timesteps==0 and restored._n_updates==0
            assert 'train' not in restored.__dict__
            for name,value in model.policy.state_dict().items():
                assert torch.equal(value,restored.policy.state_dict()[name])
            model.num_timesteps=10000
            with pytest.raises(ValueError,match='counters mismatch'):
                audit.verify(model,10000)
        finally:
            audit.close()


def test_only_fresh_matching_recurrent_learner_can_enter_loop():
    with preserved_rng():
        learner=construct('gru_sac',11)
        config=candidate_config(11)
        before=rng_record()
        validate_initial_learner(learner,config)
        assert rng_record()==before
        with pytest.raises(ValueError,match='matching portable'):
            validate_initial_learner(learner,replace(config,transitions=10000))
        learner.updates=1
        with pytest.raises(ValueError):
            validate_initial_learner(learner,config)

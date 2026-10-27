"""Local checks cannot enter actual Colab profiling or perform learner updates."""
import pytest
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
import torch

from rl_stocks.training import recurrent_comparison_profile as worker
from rl_stocks.training.recurrent_comparison_starts import load_start,build
from rl_stocks.training.distributional_critics import rng_record
from rl_stocks.training.recurrent_sac_learner import RecurrentSACLearner
from rl_stocks.training.sac_update_worker import preserved_rng


def forbid_optimization(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('local optimization attempted')
    for cls in (SAC,RecurrentPPO):
        monkeypatch.setattr(cls,'learn',forbidden)
        monkeypatch.setattr(cls,'train',forbidden)
    monkeypatch.setattr(RecurrentSACLearner,'update',forbidden)
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)


def test_profile_guard_precedes_dataset_output_models_and_rng(tmp_path,monkeypatch):
    forbid_optimization(monkeypatch)
    def forbidden(*args,**kwargs):
        pytest.fail('profile reached model construction locally')
    monkeypatch.setattr(worker,'load_start',forbidden)
    output=tmp_path/'artifacts'/'profile'
    before=rng_record()
    with pytest.raises(RuntimeError,match='Colab CPU'):
        worker.execute_colab({},tmp_path/'missing-starts',output)
    assert not output.exists() and rng_record()==before


def test_load_each_actual_start_restores_constructor_streams(tmp_path,monkeypatch):
    forbid_optimization(monkeypatch)
    directory=tmp_path/'starts'
    manifest=build(directory)
    with preserved_rng():
        for arm in ('native_sac','recurrent_ppo_reference','gru_sac'):
            model=load_start(directory,arm,11)
            assert rng_record()==manifest['entries'][arm+'-11']['rng_hashes']
            if arm=='gru_sac':
                assert model.updates==0 and model.max_updates==99000
                assert all(not opt.state for opt in model.optimizers.values())
            else:
                assert model._n_updates==0 and model.num_timesteps==0
                if arm=='native_sac':
                    assert model.batch_size==256 and model.buffer_size==100000
                    assert model.log_ent_coef.item()==0 and not model.ent_coef_optimizer.state
                else:
                    assert model.n_steps==1000 and model.batch_size==50 and model.n_epochs==10
                    assert not model.policy.optimizer.state

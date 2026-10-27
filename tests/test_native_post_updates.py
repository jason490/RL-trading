"""Native checkpoint phase and save hygiene; spies never optimize locally."""
import json
from types import SimpleNamespace
import zipfile

import pytest
from stable_baselines3 import SAC
import torch

from rl_stocks.training.native_post_updates import post_update_checkpoints, native_snapshot_view
from rl_stocks.training.crossq_inspection import SpacesOnly
from rl_stocks.training.sac_update_worker import preserved_rng


class Spy:
    num_timesteps=2
    def __init__(self):
        self.events=[]
    def train(self):
        self.events.append('update complete')
        return 'result'


def test_callback_after_native_return_and_no_duplicate_checkpoint():
    model=Spy()
    def checkpoint(value,step):
        assert 'train' not in value.__dict__
        model.events.append(('checkpoint',step))
    with post_update_checkpoints(model,every=2,checkpoint=checkpoint):
        assert model.train()=='result'
        model.train()
        model.num_timesteps=3
        model.train()
        model.num_timesteps=4
        model.train()
    assert model.events==['update complete',('checkpoint',2),'update complete',
                          'update complete','update complete',('checkpoint',4)]
    assert 'train' not in model.__dict__


def test_native_error_or_callback_error_does_not_leave_hook_attached():
    class Bad(Spy):
        def train(self):
            raise RuntimeError('spy native failure')
    model=Bad()
    called=[]
    with pytest.raises(RuntimeError):
        with post_update_checkpoints(model,every=2,checkpoint=lambda *args:called.append(args)):
            model.train()
    assert not called and 'train' not in model.__dict__
    model=Spy()
    def bad_checkpoint(*args):
        raise ValueError('callback failed')
    with pytest.raises(ValueError):
        with post_update_checkpoints(model,every=2,checkpoint=bad_checkpoint):
            model.train()
    assert 'train' not in model.__dict__


def test_untrained_native_save_excludes_live_train_closure(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('local optimization attempted')
    monkeypatch.setattr(SAC,'train',forbidden)
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
    with preserved_rng():
        env=SpacesOnly(dict(observation_size=21,action_size=4))
        model=SAC('MlpPolicy',env,seed=11,device='cpu',policy_kwargs=dict(net_arch=[64,64]))
        try:
            with post_update_checkpoints(model,every=10000,checkpoint=lambda *args:None):
                with native_snapshot_view(model):
                    model.save(tmp_path/'native.zip')
            with zipfile.ZipFile(tmp_path/'native.zip') as archive:
                assert 'train' not in json.loads(archive.read('data'))
            restored=SAC.load(tmp_path/'native.zip',device='cpu')
            assert 'train' not in restored.__dict__ and restored._n_updates==0
            for name,value in model.policy.state_dict().items():
                assert torch.equal(value,restored.policy.state_dict()[name])
        finally:
            env.close()


def test_existing_instance_override_rejects():
    model=SimpleNamespace(train=lambda:None,num_timesteps=0)
    with pytest.raises(ValueError):
        with post_update_checkpoints(model,every=1,checkpoint=lambda *args:None):
            pass

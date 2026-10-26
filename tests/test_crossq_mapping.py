from copy import deepcopy

import pytest
from sb3_contrib import CrossQ
from stable_baselines3 import SAC
import torch

from rl_stocks.training.crossq_mapping import map_linears, validate_mapping
from rl_stocks.training.crossq_spec import parameters
from rl_stocks.training.crossq_inspection import SpacesOnly


def models(seed=11):
    env = SpacesOnly(dict(observation_size=21, action_size=4))
    return (SAC('MlpPolicy', env, seed=seed, device='cpu', **parameters('sac')),
            CrossQ('MlpPolicy', env, seed=seed, device='cpu', **parameters('crossq')))


@pytest.fixture(autouse=True)
def no_optimization(monkeypatch):
    def forbidden(*a, **k):
        pytest.fail('local optimization attempted')
    for cls in [SAC, CrossQ]:
        monkeypatch.setattr(cls, 'learn', forbidden)
        monkeypatch.setattr(cls, 'train', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)


@pytest.mark.parametrize('seed', [11, 23, 37])
def test_common_linears_heads_and_critics_are_copied_without_rng_or_owner_change(seed):
    sac, crossq = models(seed)
    with torch.no_grad():
        sac.actor.mu.weight.add_(0.01)  # Distinct untrained source fixture; no fitting.
    rng = torch.get_rng_state().clone()
    owners = [id(p) for p in crossq.policy.parameters()]
    result = map_linears(sac, crossq)
    assert result['common_linears_exact'] and len(result['mapping']) == 10
    assert validate_mapping(sac, crossq) == {k: v for k, v in result.items() if k != "parameter_ownership_preserved"}
    assert result["parameter_ownership_preserved"]
    assert torch.equal(rng, torch.get_rng_state())
    assert owners == [id(p) for p in crossq.policy.parameters()]
    assert not crossq.actor.optimizer.state and not crossq.critic.optimizer.state


@pytest.mark.parametrize('change', ['normalization', 'optimizer', 'trained', 'nonfinite', 'delay', 'ownership'])
def test_bad_source_or_scaffold_rejects_before_any_target_mutation(change):
    sac, crossq = models()
    if change == 'normalization':
        crossq.actor.latent_pi[0].steps.fill_(1)
    elif change == 'optimizer':
        crossq.critic.optimizer.param_groups[0]['betas'] = (0.9, 0.999)
    elif change == 'trained':
        sac.num_timesteps = 1
    elif change == 'nonfinite':
        with torch.no_grad():
            sac.actor.mu.weight.fill_(float('nan'))
    elif change == 'ownership':
        crossq.actor.optimizer.param_groups[0]['params'] = []
    else:
        crossq.policy_delay = 1
    before = deepcopy(crossq.policy.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        map_linears(sac, crossq)
    assert all(torch.equal(before[k], v) for k, v in crossq.policy.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())

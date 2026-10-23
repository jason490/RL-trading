import math
import numpy as np
import pytest
from rl_stocks.reporting.action_boundaries import boundary_mass,recurrent_parameters


def test_standard_normal_exact_marginals_and_independent_joint():
    r=boundary_mass(np.zeros((3,4)),np.ones((3,4)));p=math.erfc(1/math.sqrt(2))
    assert r['observations']==3
    for c in r['components'].values():
        assert c['outside_probability']==pytest.approx(p)
        assert c['below_lower_probability']==pytest.approx(p/2)
        assert c['above_upper_probability']==pytest.approx(p/2)
    assert r['any_component_outside_probability']==pytest.approx(1-(1-p)**4)


def test_directional_asymmetry_and_near_deterministic_boundaries():
    means=np.array([[3.,-3.,0.,0.]])
    r=boundary_mass(means,np.full((1,4),.001))
    assert r['components']['exposure']['above_upper_probability']==1.
    assert r['components']['stop_distance']['below_lower_probability']==1.
    assert r['components']['zone_width']['outside_probability']==0.
    assert r['any_component_outside_probability']==1.
    assert np.array_equal(means,[[3.,-3.,0.,0.]])


@pytest.mark.parametrize('means,stds',[(np.zeros(4),np.ones(4)),([],[]),([[0]*3],[[1]*3]),
    ([[0]*4],[[0]*4]),([[0]*4],[[-1]*4]),([[float('nan')]*4],[[1]*4]),
    ([[0]*4],[[float('inf')]*4]),([[0]*4],[[1]*4]*2)])
def test_invalid_gaussian_contract_rejects(means,stds):
    with pytest.raises(ValueError):boundary_mass(means,stds)


def test_recurrent_prefix_state_reset_flat_account_and_model_rng_immutability():
    import torch
    from test_clipping_protocol import model
    from rl_stocks.training.input_clipping import apply_initialization,model_record
    from rl_stocks.data.generation.datasets import Normalizer,normalizer_from
    from test_exposure_replay import fixture
    from dataclasses import asdict
    from rl_stocks.features.registry.assemble import NAMES
    data=fixture();data['names']=list(NAMES);data['normalizer']=asdict(Normalizer(NAMES,(0.,)*7,(1.,)*7))
    episodes=list(data['episodes'].values())
    for e in episodes:e.update(split='train',features=[[0.]*7 for _ in e['bars']])
    m=model('control');apply_initialization(m,'control',11,True);before=model_record(m);rng=torch.get_rng_state().clone()
    result=recurrent_parameters(m,episodes,normalizer_from(data['normalizer']),data['names'],decisions=2)
    again=recurrent_parameters(m,episodes,normalizer_from(data['normalizer']),data['names'],decisions=2)
    assert result==again and result['observations']==2
    assert model_record(m)==before and torch.equal(rng,torch.get_rng_state())
    with pytest.raises(ValueError):recurrent_parameters(m,[dict(episodes[0],split='validation')],normalizer_from(data['normalizer']),data['names'])


def test_extreme_finite_variance_preserves_tail_asymmetry_without_scale_overflow():
    r=boundary_mass([[1e308]*4],[[1e308]*4])
    assert r['components']['exposure']['above_upper_probability']==pytest.approx(.8413447460685429)
    assert r['components']['exposure']['below_lower_probability']==pytest.approx(.15865525393145707)

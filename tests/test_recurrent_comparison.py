"""Fixed protocol, conservative timing arithmetic and paired synthetic gates."""
from copy import deepcopy

import pytest

from rl_stocks.training.recurrent_comparison import protocol, candidate_config, higher_p95, profile_admission, PROTOCOL_SHA256
from rl_stocks.reporting.recurrent_sac_comparison import screen_gate, confirmation_gate


def profile():
    spec=protocol()
    return dict(protocol_sha256=PROTOCOL_SHA256,dataset=spec['dataset'],device='cpu',
                default_python_family=[3,13],candidate_updates=64,native_sac_updates=64,
                recurrent_ppo_transitions=3000,recurrent_ppo_adam_steps=600,
                candidate_collection_seconds=[.001]*512,actor_refresh512_seconds=[.002]*64,
                candidate_sample_update_seconds=[.004]*64,native_sac_update_seconds=[.002]*64,
                recurrent_ppo_rollout1000_seconds=[4.,5.,6.],validation_seconds=[10.,12.,11.],
                checkpoint_io_seconds=[1.,2.,3.])


def phase(score):
    return dict(score=score,mean_max_drawdown=-.1,groups=dict(
        stock=dict(net_return=score+.02,excess_return=score-.3,median_excess_return=score-.3),
        etf=dict(net_return=score-.02,excess_return=score-.3)))


def cells(score=.35):
    return dict(native_sac=dict(selected=phase(.2),final=phase(.18)),
                recurrent_ppo_reference=dict(selected=phase(.22),final=phase(.21)),
                gru_sac=dict(selected=phase(score),final=phase(score)))


def test_fixed_matrix_information_roles_and_native_update_counts():
    spec=protocol()
    assert len(spec['training_symbols'])==111 and len(spec['validation_symbols'])==26
    assert spec['arm_contracts']['native_sac']['observation_profile']=='execution_full'
    assert spec['arm_contracts']['recurrent_ppo_reference']['observation_profile']=='execution_masked'
    assert spec['arm_contracts']['gru_sac']['observation_profile']=='execution_full'
    assert spec['checkpoint_phase']=='post_update'
    for seed in (11,23,37):
        cfg=candidate_config(seed)
        assert cfg.transitions==100000 and cfg.maximum_updates==99000 and cfg.seed==seed
    with pytest.raises(ValueError):
        candidate_config(7)
    assert spec['execution']['project_ceiling']==150
    assert spec['execution']['project_teardown']==130
    assert spec['execution']['minimum_account_balance']==150
    assert spec['execution']['account_teardown_balance']==170


def test_profile_exact_projection_p95_and_no_physical_admission_claim():
    values=profile()
    result=profile_admission(values)
    expected=1.5*(100000*(.001+.002)+99000*.004+10*12+11*3)+600+600+360
    assert result['required_allocation_seconds']['gru_sac']==pytest.approx(expected)
    assert result['profile_projection_passed']
    assert result['physical_profile_acceptance_still_required']
    assert result['performance_interpretation_allowed'] is False
    assert higher_p95([1.,2.,3.],count=3)==3
    values['candidate_sample_update_seconds'][:8]=[1000.]*8
    assert profile_admission(values)==result  # fixed warm-update exclusion, no cherry-picking


def test_slow_profile_blocks_full_budget_without_shortening_or_mutating_config():
    values=profile()
    values['candidate_sample_update_seconds']=[.1]*64
    before=deepcopy(values)
    result=profile_admission(values)
    assert not result['profile_projection_passed'] and not result['arm_passes']['gru_sac']
    assert values==before and candidate_config(11).transitions==100000


@pytest.mark.parametrize('change',['nan','short','bool','count','dataset','device','validation'])
def test_unreliable_profile_rejects(change):
    value=profile()
    if change=='nan':
        value['actor_refresh512_seconds'][0]=float('nan')
    elif change=='short':
        value['native_sac_update_seconds'].pop()
    elif change=='bool':
        value['candidate_collection_seconds'][0]=True
    elif change=='count':
        value['candidate_updates']=63
    elif change=='dataset':
        value['dataset']='0'*64
    elif change=='device':
        value['device']='cuda'
    else:
        value['validation_seconds'][0]=0
    with pytest.raises(ValueError):
        profile_admission(value)


def test_selected_only_win_and_bad_absolute_groups_cannot_pass_screen():
    assert screen_gate(cells())['passed']
    bad=cells()
    bad['gru_sac']['final']=phase(.1)
    assert not screen_gate(bad)['passed']
    bad=cells()
    bad['gru_sac']['final']['groups']['etf']['excess_return']=-.03
    assert not screen_gate(bad)['passed']
    bad=cells()
    bad['gru_sac']['selected']['groups']['stock']['median_excess_return']=-.03
    assert not screen_gate(bad)['passed']


def test_three_seed_confirmation_mean_and_paired_gates_prevent_one_seed_rescue():
    value={11:cells(),23:cells(),37:cells()}
    result=confirmation_gate(value)
    assert result['passed'] and result['development_outperformance']
    assert result['objective_achieved'] is False and result['prospective_confirmation_required']
    value={11:cells(.8),23:cells(.15),37:cells(.15)}
    assert not confirmation_gate(value)['passed']  # high aggregate mean cannot rescue two failed seeds
    value={11:cells(.35),23:cells(.35),37:cells(-.8)}
    assert not confirmation_gate(value)['passed']  # two wins cannot rescue catastrophic paired mean
    with pytest.raises(ValueError):
        confirmation_gate({11:cells(),23:cells()})


def test_comparable_holding_gate_does_not_claim_outperformance():
    result=confirmation_gate({seed:cells(.29) for seed in (11,23,37)})
    assert result['passed'] and not result['development_outperformance']


@pytest.mark.parametrize('key',['validation_seconds','checkpoint_io_seconds'])
@pytest.mark.parametrize('value',[None,3,'123',[],[True,1.,2.]])
def test_malformed_aggregate_timings_reject_before_projection(key,value):
    evidence=profile()
    evidence[key]=value
    with pytest.raises(ValueError):
        profile_admission(evidence)

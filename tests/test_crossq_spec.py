from copy import deepcopy

import pytest

from rl_stocks.training.crossq_spec import expected_updates, parameters, protocol
from rl_stocks.reporting.crossq_comparison import screen_gate


def test_fixed_same_geometry_with_explicit_native_package_differences():
    sac, candidate = parameters('sac'), parameters('crossq')
    assert sac['policy_kwargs']['net_arch'] == candidate['policy_kwargs']['net_arch'] == [64, 64]
    assert sac['policy_kwargs']['optimizer_kwargs']['betas'] == (0.9, 0.999)
    assert candidate['policy_kwargs']['optimizer_kwargs']['betas'] == (0.5, 0.999)
    assert candidate['policy_delay'] == 3 and 'tau' not in candidate
    assert sac['tau'] == 0.005 and 'batch_norm' not in sac['policy_kwargs']
    assert candidate['policy_kwargs']['renorm_warmup_steps'] == 1000
    smoke = parameters('crossq', True)
    assert smoke['policy_kwargs'] == candidate['policy_kwargs']
    assert smoke['batch_size'] == smoke['learning_starts'] == 64
    smoke['policy_kwargs']['net_arch'][0] = 999
    assert parameters('crossq', True)['policy_kwargs']['net_arch'] == [64, 64]


def test_phase_sensitive_counts_and_smoke_exercises_both_renorm_branches():
    assert expected_updates('sac', 50000, 'post_update') == dict(critic=49000, actor=49000, entropy=49000)
    assert expected_updates('sac', 50000, 'on_step')['critic'] == 48999
    row = expected_updates('crossq', 4096, 'post_update', True)
    assert row == dict(critic=4032, actor=1344, entropy=1344,
                      critic_normalization=4032, actor_normalization=1344,
                      critic_renorm_forward_calls=3031, actor_renorm_forward_calls=343)
    assert expected_updates('crossq', 1000, 'post_update')['critic'] == 0
    assert expected_updates('crossq', 50000, 'post_update')['actor'] == 16333


@pytest.mark.parametrize('arm,steps,phase,smoke', [
    ('crossq', True, 'on_step', False), ('crossq', 4097, 'post_update', True),
    ('crossq', 50001, 'post_update', False), ('crossq', 1000, 'bad', False),
    ('unknown', 1, 'post_update', False), ('sac', 1, 'post_update', 1),
])
def test_invalid_phase_or_counter_cannot_alias_valid_contract(arm, steps, phase, smoke):
    with pytest.raises(ValueError):
        expected_updates(arm, steps, phase, smoke)


def cells():
    return {arm: {phase: dict(score=score, mean_max_drawdown=-0.1,
                groups={group: dict(net_return=score, excess_return=score-0.3,
                                   median_excess_return=score-0.3) for group in ['stock', 'etf']})
                  for phase in ['selected', 'final']} for arm, score in [('sac', 0.28), ('crossq', 0.31)]}


def test_absolute_final_risk_and_historical_gates_cannot_be_bypassed():
    data = cells()
    assert screen_gate(data, dict(passed=True))['passed']
    mismatch = screen_gate(data, dict(passed=False))
    assert not mismatch['passed'] and mismatch['requires_integrity_investigation']
    assert not mismatch['failure_closes_fixed_family']
    for key in ['final_return', 'etf_excess', 'drawdown', 'median']:
        value = deepcopy(data)
        row = value['crossq']['final']
        if key == 'final_return':
            row['score'] = 0.29
        elif key == 'etf_excess':
            row['groups']['etf']['excess_return'] = -0.021
        elif key == 'drawdown':
            row['mean_max_drawdown'] = -0.121
        else:
            row['groups']['stock']['median_excess_return'] = -0.021
        assert not screen_gate(value, dict(passed=True))['passed']
    assert not screen_gate(data, dict(passed=True))['objective_achieved']
    assert protocol()['steps_per_cell'] == 50000

import numpy as np
import pytest

from rl_stocks.evaluation.soft_consistency import fingerprint, soft_target, summarize_windows


def test_first_action_entropy_excluded_future_action_and_soft_bootstrap_included():
    result = soft_target([1, 2], [1000, -3], gamma=0.5, alpha=0.2,
                         terminated=False, bootstrap_q=4, bootstrap_log_probability=-2)
    assert result['discounted_rewards'] == 2
    assert result['future_action_entropy'] == pytest.approx(0.3)
    assert result['discounted_bootstrap_q'] == 1
    assert result['bootstrap_entropy'] == pytest.approx(0.1)
    assert result['target'] == pytest.approx(3.4)


def test_one_step_matches_sac_soft_bellman_target_at_timeout():
    result = soft_target([1], [999], gamma=0.9, alpha=0.1,
                         terminated=False, bootstrap_q=2, bootstrap_log_probability=3)
    assert result['target'] == pytest.approx(1 + 0.9 * (2 - 0.1 * 3))
    assert result['future_action_entropy'] == 0


def test_true_termination_removes_both_bootstrap_terms():
    result = soft_target([1], [999], gamma=0.9, alpha=0.1,
                         terminated=True, bootstrap_q=200, bootstrap_log_probability=-300)
    assert result['target'] == 1 and result['bootstrap_discount'] == 0


@pytest.mark.parametrize('rewards,logp', [([], []), ([1], [1, 2]), ([float('nan')], [0])])
def test_bad_soft_inputs_rejected(rewards, logp):
    with pytest.raises(ValueError):
        soft_target(rewards, logp, gamma=0.99, alpha=0.1, terminated=False)


def test_optimizer_fingerprint_detects_mutation_without_touching_state():
    state = {'state': {0: {'exp_avg': np.array([1., 2.])}}, 'lr': 0.0003}
    before = fingerprint(state)
    assert before == fingerprint(state)
    state['state'][0]['exp_avg'][0] = 3
    assert before != fingerprint(state)


def test_conditional_summary_uses_window_means_not_correlated_trajectory_rows():
    result = summarize_windows([{'mean_min_q_minus_soft_target': 1},
                                {'mean_min_q_minus_soft_target': 3}])
    assert result['mean_min_q_minus_soft_target'] == 2
    assert result['conditional_window_sampling_se'] == 1
    assert result['descriptive_lower_two_se'] == 0

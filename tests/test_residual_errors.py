import numpy as np
import pytest
from rl_stocks.reporting.residual_errors import residual_effect


def test_useful_correction_and_daily_components_reconcile_without_mutation():
    y = np.array([-1.0, 1.0, -2.0, 2.0])
    baseline = np.zeros(4)
    pred = 0.5 * y
    before = [v.copy() for v in (y, baseline, pred)]
    r = residual_effect(y, baseline, pred, ["a", "a", "b", "b"])
    assert r["candidate_minus_baseline_mse"] == pytest.approx(-1.875)
    assert r["correction_second_moment"] == pytest.approx(0.625)
    assert r["baseline_error_correction_product_mean"] == pytest.approx(1.25)
    assert r["daily_common_component"] == 0
    assert r["daily_within_component"] == pytest.approx(-1.875)
    for actual, original in zip((y, baseline, pred), before):
        np.testing.assert_array_equal(actual, original)


def test_perfect_ranking_can_worsen_mse_when_correction_energy_is_too_large():
    r = residual_effect([-1.0, 1.0], [0.0, 0.0], [-3.0, 3.0], ["a", "a"])
    assert r["baseline_error_correction_covariance"] == 3
    assert r["candidate_minus_baseline_mse"] == 3
    r = residual_effect([1.0, 3.0, 5.0, 7.0], [0.0] * 4, [2.0] * 4, ["a", "a", "b", "b"])
    assert r["daily_common_component"] == -12
    assert r["daily_within_component"] == 0


@pytest.mark.parametrize(
    "y,b,p,days",
    [
        ([], [], [], []),
        ([1.0], [0.0, 1.0], [0.0], ["a"]),
        ([1.0], [0.0], [np.nan], ["a"]),
        ([1.0], [0.0], [0.0], [1]),
        ([1.0], [0.0], [0.0], []),
        ([1e308], [-1e308], [1e308], ["a"]),
    ],
)
def test_bad_or_overflowing_inputs_rejected(y, b, p, days):
    with pytest.raises(ValueError):
        residual_effect(y, b, p, days)

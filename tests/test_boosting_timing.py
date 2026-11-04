import pytest
from rl_stocks.runtime.colab.boosting_timing import projection


def test_cold_import_is_not_scaled_with_tree_count_and_reserves_remain():
    values = dict(
        cold_constructor_seconds=10.0,
        fit_seconds=0.1,
        acceptance_seconds=0.2,
        warm_constructor_seconds=0.01,
    )
    result = projection(**values)
    assert result["projected_remaining_seconds"] == pytest.approx(1360.92)
    values["cold_constructor_seconds"] = 100.0
    assert (
        projection(**values)["projected_remaining_seconds"] == result["projected_remaining_seconds"]
    )
    assert result["remaining_work_reserve_seconds"] == 900 and result["safety_multiplier"] == 2
    assert result["full_models"] == 6 and result["trees_per_model"] == 256


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
def test_invalid_measurements_rejected(value):
    with pytest.raises(ValueError):
        projection(
            cold_constructor_seconds=1.0,
            fit_seconds=value,
            acceptance_seconds=0.01,
            warm_constructor_seconds=0.01,
        )


def test_zero_fit_and_overflow_rejected():
    for fit in [0.0, 1e308]:
        with pytest.raises(ValueError):
            projection(
                cold_constructor_seconds=1.0,
                fit_seconds=fit,
                acceptance_seconds=1.0,
                warm_constructor_seconds=1.0,
            )

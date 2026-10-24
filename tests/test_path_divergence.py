from copy import deepcopy

import pytest

from rl_stocks.reporting.path_divergence import compare_paths


def episode(equities=(101000.0, 100500.0), quantities=(10, 0), reasons=("add", "stop")):
    rows, events = [], []
    previous, quantity = 100000.0, 0
    for date, equity, target, reason in zip(
        ("2021-01-04", "2021-01-05"), equities, quantities, reasons
    ):
        rows.append(
            dict(
                session=date,
                price=100.0,
                equity=equity,
                previous_equity=previous,
                quantity=target,
                approved_target_quantity=target,
                raw_action=[0.5, 0.0, 0.0, 0.0],
            )
        )
        events.append(
            dict(time=date + "T15:00:00Z", quantity=target, delta=target - quantity, reason=reason)
        )
        previous, quantity = equity, target
    return dict(
        symbol="A",
        split="validation",
        rows=rows,
        events=events,
        baselines=dict(buy_hold=dict(fixed="same")),
        metrics=dict(net_return=equities[-1] / 100000.0 - 1),
    )


def test_bridge_first_exits_and_inherited_side_reconcile():
    before = episode()
    after = episode((100800.0, 101200.0), (9, 0), ("add", "target_near"))
    result = compare_paths(before, after)
    assert result["final_minus_selected_pp"] == pytest.approx(0.7)
    assert result["first_differences"]["closing_quantity"]["session"] == "2021-01-04"
    assert result["first_differences"]["protective_reasons"]["session"] == "2021-01-05"
    assert result["difference_session_counts"]["closing_side"] == 0
    assert result["buckets"]["flat/flat/protection_same"]["net_difference_pp"] == -0.2
    assert result["buckets"]["long/long/protection_different"]["net_difference_pp"] == 0.9
    assert sum(v["net_difference_pp"] for v in result["buckets"].values()) == pytest.approx(0.7)


def test_identical_paths_and_side_difference():
    before = episode()
    assert compare_paths(before, deepcopy(before))["final_minus_selected_pp"] == 0
    after = episode(quantities=(-10, 0))
    result = compare_paths(before, after)
    assert result["first_differences"]["closing_side"]["session"] == "2021-01-04"
    assert result["difference_session_counts"]["closing_side"] == 1


@pytest.mark.parametrize(
    "mutation", ["price", "date", "fill", "equity", "endpoint", "baseline", "action"]
)
def test_bad_pair_rejected(mutation):
    before, after = episode(), episode()
    if mutation == "price":
        after["rows"][0]["price"] = 101.0
    elif mutation == "date":
        after["rows"][0]["session"] = "2024-01-04"
    elif mutation == "fill":
        after["events"][0]["delta"] = 9
    elif mutation == "equity":
        after["rows"][1]["previous_equity"] = 1.0
    elif mutation == "endpoint":
        after["metrics"]["net_return"] = 0.5
    elif mutation == "baseline":
        after["baselines"]["buy_hold"] = {}
    else:
        after["rows"][0]["raw_action"][0] = float("nan")
    with pytest.raises(ValueError):
        compare_paths(before, after)

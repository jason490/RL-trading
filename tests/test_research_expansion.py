from collections import Counter
from dataclasses import replace
from datetime import date, timedelta
import importlib.util
import json
import math
from pathlib import Path

import pytest

from rl_stocks.core.contracts import Bar
from rl_stocks.data.universe.config import read_universe
from rl_stocks.features.market.relationships import CrossStockRelationships


def test_expansion_and_separate_nontradable_indices():
    root = Path(__file__).resolve().parents[1]
    parent = read_universe(root / "configs/universes/stock_research_110.json")
    expanded = read_universe(root / "configs/universes/stock_research_132.json")
    old = {r["symbol"] for r in parent["instruments"]}
    added = [r for r in expanded["instruments"] if r["symbol"] not in old]
    assert len(added) == 22
    assert set(Counter(r["sector"] for r in added).values()) == {2}
    assert sum(r["kind"] == "stock" for r in expanded["instruments"]) == 132
    for name in ("vix", "move"):
        context = json.loads((root / f"configs/context/{name}.json").read_text())
        assert context["role"] == "context" and context["tradable"] is False
        assert context["model_inputs_enabled"] is False
        assert context["symbol"] not in {r["symbol"] for r in expanded["instruments"]}


def bars():
    rows = []
    for symbol, direction in (("AAA", 1), ("SPY", 1), ("XLK", -1)):
        price = 100
        for i in range(24):
            price *= math.exp(direction * (i % 3 + 1) / 100)
            day = str(date(2020, 1, 1) + timedelta(days=i))
            rows.append(Bar(symbol, day, day + "T14:30:00+00:00", day + "T21:00:00+00:00",
                            day + "T21:15:00+00:00", day + "T21:15:00+00:00",
                            "synthetic", "synthetic", "v1", price, price, price, price, 100))
    return rows


def test_relationship_alignment_availability_and_future_isolation():
    rows = bars()
    missing = dict.fromkeys(("AAA", "SPY", "XLK"), [])
    feature = CrossStockRelationships(rows, missing, {"AAA": "XLK"})
    value = feature.at("AAA", "2020-01-21", "2020-01-21T21:15:00+00:00")
    assert value[:2] == pytest.approx([1, -1])
    assert value[3] > 0 and value[2] < 0
    assert feature.at("AAA", "2020-01-21", "2020-01-21T21:14:00+00:00") is None
    changed = [replace(b, close=b.close * 2, high=b.high * 2)
               if b.session > "2020-01-21" else b for b in rows]
    other = CrossStockRelationships(changed, missing, {"AAA": "XLK"})
    assert other.at("AAA", "2020-01-21", "2020-01-21T21:15:00+00:00") == value
    missing["XLK"] = ["2020-01-10"]
    assert CrossStockRelationships(rows, missing, {"AAA": "XLK"}).at(
        "AAA", "2020-01-21", "2020-01-21T21:15:00+00:00") is None


def test_move_rejects_executable_or_retired_request_before_network():
    path = Path(__file__).resolve().parents[1] / "scripts/collect_research_expansion.py"
    spec = importlib.util.spec_from_file_location("expansion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="nontradable"):
        module.move_snapshot({"symbol": "^MOVE", "role": "tradable", "tradable": True})

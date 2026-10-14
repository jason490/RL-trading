from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
import tempfile
import unittest

from rl_stocks.core.contracts import Bar, Request
from rl_stocks.data.providers.synthetic.adapter import SyntheticProvider
from rl_stocks.features.market_structure.supply_demand import ZoneConfig, ZoneEngine, supply_demand
from rl_stocks.features.registry.assemble import assemble


def candle(day, op=100, high=101, low=99, close=100):
    return Bar("SPY", day, day+"T14:30:00+00:00", day+"T21:00:00+00:00",
               day+"T21:15:00+00:00", "2026-10-14T22:00:00+00:00",
               "synthetic", "synthetic", "fixture", op, high, low, close, 1000)


class SupplyDemandTests(unittest.TestCase):
    def engine(self, short=False):
        engine = ZoneEngine(config=ZoneConfig(atr_period=2))
        for day in ("2020-01-02", "2020-01-03"):
            engine.update(candle(day))
        engine.update(candle("2020-01-06", 100, 101, 95, 96) if short else
                      candle("2020-01-06", 100, 105, 99, 104))
        self.assertEqual(len(engine.zones), 1)
        return engine

    def test_both_directions_and_break_priority(self):
        for short in (False, True):
            engine = self.engine(short)
            zone = engine.zones[0]
            self.assertEqual(zone.kind, "supply" if short else "demand")
            self.assertEqual((zone.bottom, zone.top), (99, 101))
            engine.update(candle("2020-01-07", 100, 103, 97, 102 if short else 98))
            self.assertEqual(zone.state, "broken")
            self.assertEqual(engine.events[-1]["state"], "broken")

    def test_mitigation_does_not_rewrite_emitted_snapshot(self):
        engine = self.engine()
        sessions = [(b.session, b.available_time) for b in engine.bars]
        vector, snapshot = engine.snapshot(104, 2, "2020-01-06", sessions[-1][1], sessions)
        self.assertEqual(snapshot["demand"]["state"], "fresh")
        engine.update(candle("2020-01-07", 103, 104, 100, 102))
        self.assertEqual(engine.zones[0].state, "mitigated")
        self.assertEqual(snapshot["demand"]["state"], "fresh")
        self.assertEqual(vector[4], 1.)
        with self.assertRaises(ValueError):
            engine.snapshot(104, 2, "2020-01-06", sessions[-1][1], sessions)

    def test_readiness_absence_and_config_rejection(self):
        engine = ZoneEngine(config=ZoneConfig(atr_period=2))
        vector, _ = engine.snapshot(100, 2, "2020-01-02", "2020-01-02T21:15:00+00:00", [])
        self.assertEqual(vector, [0.]*13)
        for kwargs in ({"atr_period": 0}, {"impulse_atr": float("nan")}, {"max_base": 2.5}):
            with self.assertRaises(ValueError):
                ZoneConfig(**kwargs)
        with self.assertRaises(ValueError):
            supply_demand([], ("h1",))

    def test_future_suffix_prefix_and_completed_week_release(self):
        bars = SyntheticProvider().bars(Request(("SPY",), date(2020, 1, 1), date(2021, 1, 1)))
        full, _ = supply_demand(bars)
        for cutoff in (75, 101, 142, 221):
            prefix, _ = supply_demand(bars[:cutoff])
            self.assertEqual(prefix, full[:cutoff])
        for i, item in enumerate(full):
            for zone in item["zones"]["w1"].values():
                formation = date.fromisoformat(zone["formation_session"]).isocalendar()[:2]
                decision = date.fromisoformat(bars[i].session).isocalendar()[:2]
                self.assertLess(formation, decision)
                self.assertLessEqual(zone["available_time"], bars[i].available_time)

    def test_weekly_impulse_appears_only_next_week(self):
        from datetime import timedelta
        from rl_stocks.features.timeframes.aggregation import completed_week_bars
        start = date(2020, 1, 6)
        bars = []
        for week in range(18):
            for day in range(5):
                session = str(start+timedelta(weeks=week, days=day))
                bars.append(candle(session, 100, 106, 99, 105) if week == 15 else
                            candle(session))
        snapshots, events = supply_demand(bars, ("w1",))
        self.assertFalse(snapshots[79]["zones"]["w1"])
        self.assertIn("demand", snapshots[80]["zones"]["w1"])
        zone = snapshots[80]["zones"]["w1"]["demand"]
        self.assertEqual(zone["available_time"], bars[80].available_time)
        self.assertEqual(zone["formation_event_time"], bars[79].event_time)
        aggregate = list(completed_week_bars(bars))[80]
        self.assertEqual((aggregate.open, aggregate.high, aggregate.low, aggregate.close), (100, 106, 99, 105))
        self.assertEqual(aggregate.volume, 5000)

    def test_split_continuity_and_availability_rejection(self):
        bars = SyntheticProvider().bars(Request(("SPY",), date(2020, 1, 1), date(2020, 7, 1)))
        adjusted = bars[:80]+[replace(b, open=b.open/2, high=b.high/2, low=b.low/2,
                    close=b.close/2, volume=b.volume*2, split=2 if i == 0 else 1)
                    for i, b in enumerate(bars[80:])]
        original, names = assemble(bars, zone_timeframes=("d1", "w1"))
        split, _ = assemble(adjusted, zone_timeframes=("d1", "w1"))
        for a, b in zip(original, split):
            for x, y in zip(a["values"], b["values"]):
                self.assertAlmostEqual(x, y)
        delayed = [replace(bars[0], available_time=bars[-1].available_time), *bars[1:]]
        with self.assertRaises(ValueError):
            supply_demand(delayed)

    def test_dataset_schema_snapshots_and_train_only_normalization(self):
        from rl_stocks.data.storage.immutable import put
        from rl_stocks.data.validation.bars import coverage
        from rl_stocks.data.generation.datasets import generate, read_dataset, Normalizer
        request = Request(start=date(2020, 1, 1))
        bars = SyntheticProvider().bars(request)
        provenance = {"coverage": coverage(bars, request), "dividends_verified": True,
                      "capabilities": {"feed": "synthetic"}}
        with tempfile.TemporaryDirectory() as directory:
            snapshot = put(Path(directory)/"raw", bars, provenance)
            dataset = read_dataset(generate(snapshot, Path(directory)/"datasets",
                allow_unverified=True, zone_timeframes=("d1", "w1")))
            self.assertEqual(len(dataset["names"]), 33)
            self.assertIn("sd-v1-d1-w1", dataset["schema"])
            train = []
            for episode in dataset["episodes"].values():
                self.assertEqual(len(episode["zone_snapshots"]), len(episode["features"]))
                if episode["split"] == "train":
                    train.extend(episode["features"])
            expected = Normalizer.fit(dataset["names"], train)
            self.assertEqual(list(expected.means), dataset["normalizer"]["means"])

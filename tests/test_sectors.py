from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest

from rl_stocks.core.contracts import Request
from rl_stocks.data.providers.synthetic.adapter import SyntheticProvider
from rl_stocks.data.universe.config import SECTORS, read_universe, symbols
from rl_stocks.data.universe.sectors import put_sectors, read_sectors, sector_asof, validate_records
from rl_stocks.features.sectors.rotation import SectorRotation


def record(symbol="AAPL", sector="technology", available="2019-01-01T00:00:00+00:00",
           source="audited-file"):
    return {"symbol": symbol, "sector": sector, "available_time": available,
            "retrieved_time": "2026-10-14T22:00:00+00:00", "effective_from": available[:10],
            "effective_to": None, "source": source, "revision": "fixture-1"}


class SectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.universe = read_universe("configs/universes/sector_research.json")
        cls.proxies = cls.universe["sector_proxies"]
        cls.bars = SyntheticProvider().bars(Request(("SPY", "AAPL", *cls.proxies.values()),
                                            date(2020, 1, 1), date(2020, 7, 1)))

    def test_configured_universe_all_stocks_have_sectors(self):
        self.assertEqual(len(symbols(self.universe, "tradable")), 26)
        self.assertEqual(len(symbols(self.universe, "context")), 11)
        stocks = [x for x in self.universe["instruments"] if x["kind"] == "stock"]
        self.assertEqual(len(stocks), 22)
        self.assertEqual({s: sum(x["sector"] == s for x in stocks) for s in SECTORS},
                         {s: 2 for s in SECTORS})

    def test_universe_rejects_missing_stock_sector_and_bad_proxy(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"universe.json"
            for field in ("sector", "role"):
                config = json.loads(json.dumps(self.universe))
                config["instruments"][4][field] = "invalid"
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    read_universe(path)

    def test_current_sector_cannot_leak_or_be_backdated(self):
        current = record(available="2026-10-14T12:00:00+00:00", source="yahoo-current-info")
        validate_records([current])
        self.assertIsNone(sector_asof([current], "AAPL", "2020-01-02", "2020-01-02T21:15:00+00:00"))
        self.assertEqual(sector_asof([current], "AAPL", "2026-10-14", "2026-10-14T21:15:00+00:00"), "technology")
        with self.assertRaises(ValueError):
            validate_records([{**current, "effective_from": "2010-01-01"}])

    def test_historical_effective_interval_and_revision(self):
        old = {**record(), "effective_to": "2020-06-01"}
        new = {**record(sector="communication_services", available="2020-06-01T20:00:00+00:00"),
               "revision": "fixture-2"}
        self.assertEqual(sector_asof([old, new], "AAPL", "2020-05-29", "2020-05-29T21:15:00+00:00"), "technology")
        self.assertIsNone(sector_asof([old, new], "AAPL", "2020-06-01", "2020-06-01T19:00:00+00:00"))
        self.assertEqual(sector_asof([old, new], "AAPL", "2020-06-01", "2020-06-01T21:15:00+00:00"), "communication_services")

    def test_immutable_sector_snapshot_and_yahoo_mapping(self):
        from rl_stocks.data.providers.yahoo.sectors import YahooSectors
        from unittest.mock import patch
        class Ticker:
            def __init__(self, symbol): pass
            def get_info(self): return {"sector": "Technology", "irrelevant": "omitted"}
        with patch("rl_stocks.data.providers.yahoo.sectors.utcnow", return_value="2026-10-14T00:00:00+00:00"):
            records = YahooSectors().records(["AAPL"], Ticker)
        self.assertEqual(records[0]["sector"], "technology")
        self.assertNotIn("irrelevant", records[0])
        with tempfile.TemporaryDirectory() as d:
            path = put_sectors(records, d)
            self.assertEqual(records, read_sectors(path))
            self.assertEqual(path, put_sectors(records, d))
            (path/"sectors.json").write_text("{}")
            with self.assertRaises(ValueError): read_sectors(path)

    def test_rotation_causal_rank_and_sector_relative_signal(self):
        context = SectorRotation(self.bars, self.proxies, records=[record()])
        stock = [b for b in self.bars if b.symbol == "AAPL"]
        b = stock[85]
        vector = context.at("AAPL", b.session, b.available_time)
        self.assertEqual(len(vector), 68)
        self.assertEqual(sum(vector[i] for i in range(0, 55, 5)), 11)
        self.assertEqual(vector[55], 1.)
        self.assertAlmostEqual(sum(vector[i] for i in range(4, 55, 5)), 5.5)
        prefix = [x for x in self.bars if x.session <= b.session]
        self.assertEqual(vector, SectorRotation(prefix, self.proxies, records=[record()]).at(
            "AAPL", b.session, b.available_time))
        future = [record(available="2026-10-14T00:00:00+00:00")]
        masked = SectorRotation(self.bars, self.proxies, records=future).at("AAPL", b.session, b.available_time)
        self.assertEqual(masked[55:], [0.]*13)

    def test_delayed_context_gaps_and_inception_are_masked(self):
        stock = [b for b in self.bars if b.symbol == "AAPL"]
        b = stock[85]
        proxy = self.proxies[SECTORS[0]]
        changed = [replace(x, available_time="2021-01-01T00:00:00+00:00")
                   if x.symbol == proxy and x.session == b.session else x for x in self.bars]
        self.assertEqual(SectorRotation(changed, self.proxies).at("AAPL", b.session, b.available_time)[:5], [0.]*5)
        gaps = {proxy: [stock[70].session]}
        self.assertEqual(SectorRotation(self.bars, self.proxies, missing=gaps).at("AAPL", b.session, b.available_time)[:5], [0.]*5)
        late = [x for x in self.bars if x.symbol != proxy or x.session >= stock[50].session]
        self.assertEqual(SectorRotation(late, self.proxies).at("AAPL", b.session, b.available_time)[:5], [0.]*5)

    def test_forward_split_and_cash_dividend_return(self):
        b = [x for x in self.bars if x.symbol == "AAPL"][85]
        split_day = [x for x in self.bars if x.symbol == "AAPL"][75].session
        changed = [replace(x, close=x.close/2, split=2 if x.session == split_day else 1)
                   if x.symbol == "AAPL" and x.session >= split_day else x for x in self.bars]
        original = SectorRotation(self.bars, self.proxies, records=[record()]).at("AAPL", b.session, b.available_time)
        actual = SectorRotation(changed, self.proxies, records=[record()]).at("AAPL", b.session, b.available_time)
        for x, y in zip(original, actual): self.assertAlmostEqual(x, y)
        dividend = [replace(x, dividend=2.) if x.symbol == "AAPL" and x.session == b.session
                    else x for x in self.bars]
        total_return = SectorRotation(dividend, self.proxies, records=[record()]).at(
            "AAPL", b.session, b.available_time)
        import math
        self.assertAlmostEqual(total_return[-1]-original[-1], math.log((b.close+2)/b.close))

    def test_masks_keep_identity_normalization(self):
        from rl_stocks.data.generation.datasets import Normalizer
        names = ["stock_sector_known", "stock_sector_technology", "sd_d1_demand_present", "sector_energy_ready"]
        normalizer = Normalizer.fit(names, [[0.]*4, [0.]*4])
        self.assertEqual(normalizer.transform([1.]*4, names), [1.]*4)

    def test_expanded_dataset_roles_schema_and_training_normalization(self):
        from rl_stocks.data.accumulation.backfill import accumulate
        from rl_stocks.data.generation.datasets import generate, read_dataset, Normalizer
        from rl_stocks.data.universe.config import history_starts
        # Shorter train interval keeps this integration fixture inexpensive.
        request = Request(symbols(self.universe), date(2020, 1, 1), date(2024, 7, 1))
        records = [record(x["symbol"]) for x in self.universe["instruments"] if x["kind"] == "stock"]
        with tempfile.TemporaryDirectory() as d:
            snapshot = accumulate(SyntheticProvider(), request, Path(d)/"raw", verify_calendar=False,
                                  history_starts=history_starts(self.universe), full_range=True)
            with self.assertRaises(ValueError):
                generate(snapshot, Path(d)/"datasets", allow_unverified=True,
                         universe=self.universe, sector_rotation=True)
            path = generate(snapshot, Path(d)/"datasets", allow_unverified=True,
                universe=self.universe, sector_rotation=True, sector_records=records,
                zone_timeframes=("d1", "w1"), sector_labels=True)
            dataset = read_dataset(path)
            self.assertEqual(len(dataset["names"]), 101)
            self.assertEqual(len(dataset["manifest"]["universe"]), 26)
            self.assertFalse(any(e["symbol"] in self.proxies.values() for e in dataset["episodes"].values()))
            train = [row for e in dataset["episodes"].values() if e["split"] == "train" for row in e["features"]]
            self.assertEqual(list(Normalizer.fit(dataset["names"], train).means), dataset["normalizer"]["means"])
            current = [record(x["symbol"], available="2026-10-14T00:00:00+00:00")
                       for x in self.universe["instruments"] if x["kind"] == "stock"]
            with self.assertRaisesRegex(ValueError, "no known training-era sector"):
                generate(snapshot, Path(d)/"datasets", allow_unverified=True,
                         universe=self.universe, sector_rotation=True, sector_records=current,
                         sector_labels=True)

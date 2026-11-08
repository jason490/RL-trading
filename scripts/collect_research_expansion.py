"""Collect new stock histories and MOVE separately; never fit or alter old datasets."""

import argparse
from dataclasses import asdict
from datetime import date, timedelta
import json
from pathlib import Path

from rl_stocks.core.contracts import Request, canonical, digest, utcnow
from rl_stocks.core.files import write_once
from rl_stocks.data.providers.yahoo.adapter import YahooProvider
from rl_stocks.data.storage.immutable import put
from rl_stocks.data.universe.config import read_universe


def additions(expanded, parent):
    old = {r["symbol"] for r in parent["instruments"]}
    return [r for r in expanded["instruments"] if r["symbol"] not in old]


def move_snapshot(spec):
    import math
    import yfinance as yf

    if (spec["symbol"] != "^MOVE" or spec["role"] != "context"
            or spec["tradable"] is not False or spec["end_exclusive"] != "2024-01-01"):
        raise ValueError("nontradable retired-free MOVE request required")
    frame = yf.Ticker(spec["symbol"]).history(
        start=spec["start"], end=spec["end_exclusive"], interval="1d",
        auto_adjust=False, actions=False, repair=False, raise_errors=True, timeout=30)
    rows = []
    for index, row in frame.iterrows():
        day, close = index.date(), float(row.Close)
        if not date.fromisoformat(spec["start"]) <= day < date(2024, 1, 1):
            raise ValueError("MOVE source outside declared historical interval")
        if not math.isfinite(close) or close <= 0:
            raise ValueError("finite positive MOVE close required")
        rows.append({"session": str(day), "close": close,
                     "available_time": str(day + timedelta(days=1)) + "T23:59:59+00:00"})
    if not rows or [r["session"] for r in rows] != sorted({r["session"] for r in rows}):
        raise ValueError("nonempty ordered unique MOVE history required")
    return {"schema": "move-context-snapshot-v1", "request": spec,
            "retrieved_time": utcnow(), "rows": rows, "model_inputs_enabled": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expanded = read_universe("configs/universes/stock_research_132.json")
    parent = read_universe(expanded["parent_universe"])
    claim = args.output.with_suffix(".claim.json")
    if args.output.exists():
        raise ValueError("fresh acquisition output required")
    write_once(claim, canonical({"started": utcnow(), "automatic_retry": False}))
    import yfinance as yf
    yf.set_tz_cache_location(".runtime/yahoo-cache")
    result = {"schema": "stock-move-expansion-acquisition-v1", "stocks": [],
              "training_performed": False, "model_inputs_enabled": False,
              "retired_test_access": False, "colab_allocated": False}
    try:
        for item in additions(expanded, parent):
            record = {"symbol": item["symbol"], "sector": item["sector"]}
            try:
                provider = YahooProvider()
                request = Request((item["symbol"],), date(2010, 1, 1), date(2024, 1, 1))
                bars = list(provider.bars(request))
                if any(not "2010-01-01" <= b.session < "2024-01-01" for b in bars):
                    raise ValueError("stock data outside declared historical interval")
                path = put("data/raw/stock-expansion-2026-11-08", bars,
                           {"capabilities": asdict(provider.capabilities),
                            "request": {"start": str(request.start), "end_exclusive": str(request.end)},
                            "sector_annotation": item["sector"], "historical_sector_verified": False},
                           provider.raw_source)
                record.update(status="retrieved", path=str(path), rows=len(bars),
                              first=bars[0].session, last=bars[-1].session)
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__, error=str(error)[:300])
            result["stocks"].append(record)
            print(record["symbol"], record["status"], flush=True)
            args.output.write_bytes(canonical(result))
        try:
            source = move_snapshot(json.loads(Path("configs/context/move.json").read_text()))
            target = Path("data/context/move") / digest(source) / "snapshot.json"
            write_once(target, canonical(source))
            result["move"] = {"status": "retrieved", "path": str(target),
                              "rows": len(source["rows"]), "first": source["rows"][0]["session"],
                              "last": source["rows"][-1]["session"]}
        except Exception as error:
            result["move"] = {"status": "failed", "error_type": type(error).__name__,
                              "error": str(error)[:300]}
    finally:
        args.output.write_bytes(canonical(result))
    print("Retrieved stocks:", sum(r["status"] == "retrieved" for r in result["stocks"]),
          "MOVE:", result.get("move", {}).get("status"), flush=True)


if __name__ == "__main__":
    main()

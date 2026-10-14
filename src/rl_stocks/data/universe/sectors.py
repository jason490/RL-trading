"""Validated immutable sector records and backward availability joins."""
from datetime import date
import json
from pathlib import Path
from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.files import write_once
from rl_stocks.data.validation.bars import timestamp
from rl_stocks.data.universe.config import SECTORS


def validate_records(records):
    seen = set()
    for r in records:
        if not r.get("symbol") or r["sector"] not in SECTORS:
            raise ValueError("invalid sector record")
        available = timestamp(r["available_time"])
        if timestamp(r["retrieved_time"]) < available:
            raise ValueError("retrieval cannot precede public availability")
        start = date.fromisoformat(r["effective_from"])
        end = date.fromisoformat(r["effective_to"]) if r.get("effective_to") else None
        if end is not None and end <= start:
            raise ValueError("invalid effective sector interval")
        if not r.get("source") or not r.get("revision"):
            raise ValueError("sector provenance required")
        if r["source"] == "yahoo-current-info" and start < available.date():
            raise ValueError("current sector snapshot cannot be backdated")
        identity = (r["symbol"], r["available_time"], r["effective_from"], r["revision"])
        if identity in seen:
            raise ValueError("duplicate sector record")
        seen.add(identity)


def put_sectors(records, output):
    validate_records(records)
    value = {"version": 1, "records": sorted(records, key=lambda r:
             (r["symbol"], r["available_time"], r["effective_from"], r["revision"]))}
    target = Path(output)/digest(value)
    target.mkdir(parents=True, exist_ok=True)
    write_once(target/"sectors.json", canonical(value))
    return target


def read_sectors(path):
    path = Path(path)
    value = json.loads((path/"sectors.json").read_text())
    if value.get("version") != 1 or digest(value) != path.name:
        raise ValueError("sector snapshot identity mismatch")
    validate_records(value["records"])
    return value["records"]


def sector_asof(records, symbol, session, decision_time):
    decision = timestamp(decision_time)
    candidates = [r for r in records if r["symbol"] == symbol and
                  timestamp(r["available_time"]) <= decision and r["effective_from"] <= session
                  and (not r.get("effective_to") or session < r["effective_to"])]
    if not candidates:
        return None
    candidates.sort(key=lambda r: (timestamp(r["available_time"]), r["effective_from"], r["revision"]))
    return candidates[-1]["sector"]

"""Explicit tradable/context universes; no implicit widening of the baseline."""
import json
from pathlib import Path
from datetime import date

SECTORS = ("communication_services", "consumer_discretionary", "consumer_staples",
           "energy", "financials", "health_care", "industrials", "materials",
           "real_estate", "technology", "utilities")


def read_universe(path):
    value = json.loads(Path(path).read_text())
    if value.get("version") != 1:
        raise ValueError("unsupported universe version")
    instruments = value.get("instruments", [])
    symbols = [x["symbol"] for x in instruments]
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("empty or duplicate universe")
    for item in instruments:
        if item.get("kind") not in ("stock", "broad_etf", "sector_etf"):
            raise ValueError("invalid instrument kind")
        if item.get("role") not in ("tradable", "context"):
            raise ValueError("invalid instrument role")
        if item["kind"] != "broad_etf" and item.get("sector") not in SECTORS:
            raise ValueError("every stock/sector ETF requires a sector")
        if item["kind"] == "broad_etf" and item.get("sector") != "diversified":
            raise ValueError("broad ETFs must be diversified")
        if "history_start" in item:
            date.fromisoformat(item["history_start"])
    if not any(x["role"] == "tradable" for x in instruments):
        raise ValueError("tradable symbols required")
    proxies = value.get("sector_proxies", {})
    if proxies:
        by_symbol = {x["symbol"]: x for x in instruments}
        if set(proxies) != set(SECTORS) or len(set(proxies.values())) != len(SECTORS):
            raise ValueError("one proxy required for each sector")
        for sector, symbol in proxies.items():
            item = by_symbol.get(symbol, {})
            if item.get("kind") != "sector_etf" or item.get("sector") != sector:
                raise ValueError("proxy must refer to a matching configured sector ETF")
        if value.get("benchmark") not in by_symbol:
            raise ValueError("benchmark absent from universe")
    return value


def symbols(value, role=None):
    return tuple(x["symbol"] for x in value["instruments"]
                 if role is None or x["role"] == role)


def history_starts(value):
    return {x["symbol"]: x["history_start"] for x in value["instruments"]
            if "history_start" in x}

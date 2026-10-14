from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
import json
import math
from statistics import mean, pstdev
from rl_stocks.core.contracts import canonical, digest, UNIVERSE
from rl_stocks.data.storage.immutable import load
from rl_stocks.features.registry.assemble import assemble, FEATURE_SCHEMA

@dataclass(frozen=True)
class Normalizer:
    names: tuple[str,...]
    means: tuple[float,...]
    scales: tuple[float,...]
    def __post_init__(self):
        if not self.names or len(set(self.names))!=len(self.names) or not len(self.names)==len(self.means)==len(self.scales):
            raise ValueError("invalid normalizer shape")
        if not all(math.isfinite(x) for x in self.means) or not all(math.isfinite(x) and x>0 for x in self.scales):
            raise ValueError("invalid normalizer parameters")
    @classmethod
    def fit(cls, names, rows):
        if not rows:
            raise ValueError("no training observations")
        if any(len(r)!=len(names) or not all(math.isfinite(v) for v in r) for r in rows):
            raise ValueError("invalid training feature matrix")
        cols = list(zip(*rows))
        masks = [n.endswith(("_ready", "_present", "_fresh")) or
                 n.startswith("stock_sector_") and n != "stock_sector_relative_20" for n in names]
        return cls(tuple(names), tuple(0. if mask else mean(c) for c, mask in zip(cols, masks)),
                   tuple(1. if mask else max(pstdev(c),1e-8) for c, mask in zip(cols, masks)))
    def transform(self, values, names):
        if tuple(names)!=self.names or len(values)!=len(self.names):
            raise ValueError("feature schema mismatch")
        if not all(math.isfinite(v) for v in values):
            raise ValueError("nonfinite observation")
        return [(v-m)/s for v,m,s in zip(values,self.means,self.scales)]

def split_name(session):
    return "train" if session <= "2020-12-31" else "validation" if session <= "2023-12-31" else "test"

def generate(snapshot, output, weekly=False, purge=20, allow_unverified=False,
             zone_timeframes=(), zone_config=None, universe=None, sector_rotation=False,
             sector_records=(), sector_labels=False):
    if purge < 20:
        raise ValueError("purge must cover maximum holding horizon (20 sessions)")
    bars, provenance = load(snapshot)
    tradable = UNIVERSE
    rotation = None
    if sector_labels and not sector_rotation:
        raise ValueError("sector labels require sector rotation")
    if universe is not None:
        from rl_stocks.data.universe.config import symbols
        tradable = symbols(universe, "tradable")
        missing_symbols = set(symbols(universe))-{b.symbol for b in bars}
        if missing_symbols:
            raise ValueError(f"universe absent from snapshot: {sorted(missing_symbols)}")
    if sector_rotation:
        if universe is None:
            raise ValueError("sector rotation requires a configured universe")
        from rl_stocks.features.sectors.rotation import SectorRotation
        from rl_stocks.data.universe.sectors import validate_records
        validate_records(sector_records)
        stocks = {x["symbol"] for x in universe["instruments"] if x["kind"] == "stock"}
        if not stocks <= {r["symbol"] for r in sector_records}:
            raise ValueError("sector metadata required for every stock; use ingest-sectors or audited records")
        rotation = SectorRotation(bars, universe["sector_proxies"],
            universe.get("benchmark", "SPY"), sector_records,
            {s: c["missing"] for s, c in provenance["coverage"].items()},
            use_classifications=sector_labels)
    if not allow_unverified and (not provenance.get("dividends_verified") or
        any(not x["calendar_verified"] or x["missing"] for x in provenance["coverage"].values())):
        raise ValueError("coverage/corporate actions not verified; audit or explicit research override required")
    episodes, training = {}, []
    names = None
    for symbol in tradable:
        series = [b for b in bars if b.symbol==symbol]
        if len(series)<80:
            raise ValueError(f"insufficient coverage for {symbol}")
        missing=provenance["coverage"][symbol]["missing"]
        starts=[0]+[i for i in range(1,len(series)) if any(series[i-1].session<day<series[i].session for day in missing)]+[len(series)]
        rows=[]
        for start,end in zip(starts,starts[1:]):
            chunk,names=assemble(series[start:end],weekly,zone_timeframes,zone_config)
            rows.extend([{**r,"bar":r["bar"]+start} for r in chunk])
        if rotation:
            names += rotation.names
            for row in rows:
                bar = series[row["bar"]]
                row["values"] += rotation.at(symbol, bar.session, bar.available_time)
        for split in ("train","validation","test"):
            selected = [r for r in rows if split_name(series[r["bar"]].session)==split]
            if split != "test":
                selected = selected[:-purge]
            # Never step across missing feature/session rows; split into contiguous segments.
            segments, segment = [], []
            for r in selected:
                gap=segment and any(series[segment[-1]["bar"]].session < day < series[r["bar"]].session for day in missing)
                if segment and (r["bar"] != segment[-1]["bar"]+1 or gap):
                    segments.append(segment); segment=[]
                segment.append(r)
            if segment:
                segments.append(segment)
            for k,segment in enumerate(segments):
                if len(segment)<2:
                    continue
                episode = {"symbol":symbol, "split":split,
                    "bars":[asdict(series[r["bar"]]) for r in segment],
                    "features":[r["values"] for r in segment],
                    "atr":[r["atr"] for r in segment]}
                if zone_timeframes:
                    episode["zone_snapshots"] = [r["zones"] for r in segment]
                    episode["zone_events"] = [event for r in segment for event in r["zone_events"]]
                episodes[f"{symbol}-{split}-{k}"] = episode
                if split=="train":
                    training.extend(episode["features"])
    if any(not any(e["split"]==s and e["symbol"]==symbol for e in episodes.values())
           for symbol in tradable for s in ("train","validation","test")):
        raise ValueError("all symbols must have all three splits")
    if sector_labels:
        known_index = names.index("stock_sector_known")
        for symbol in stocks:
            if not any(row[known_index] for e in episodes.values()
                       if e["symbol"] == symbol and e["split"] == "train" for row in e["features"]):
                raise ValueError(f"no known training-era sector for {symbol}; disable labels or provide historical records")
    normalizer = Normalizer.fit(names,training)
    payload = {"schema":FEATURE_SCHEMA+("-weekly" if weekly else ""), "names":names,
        "normalizer":asdict(normalizer), "episodes":episodes,
        "manifest":{"snapshot":Path(snapshot).name,"provenance":provenance,"universe":tradable,
            "splits":{"train_end":"2020-12-31","validation_end":"2023-12-31","test_start":"2024-01-01"},
            "purge_sessions":purge,"max_holding_sessions":20,"weekly":weekly,
            "unverified_override":allow_unverified,"survivorship":"Fixed ETF universe chosen retrospectively; no broad-universe claims"}}
    if universe is not None:
        payload["manifest"]["universe_config"] = universe
        if sector_records:
            from rl_stocks.data.universe.sectors import validate_records
            validate_records(sector_records)
            payload["manifest"]["sector_metadata"] = list(sector_records)
        payload["manifest"]["survivorship"] = (
            "Retrospectively selected surviving stocks/ETFs; not historical index membership; "
            "current declared sector labels are annotations, not historical facts")
    if sector_rotation:
        payload["schema"] += "-sector-rotation-v1"+("-labels" if sector_labels else "")
        payload["manifest"]["sector_rotation"] = {
            "version": 1, "proxies": universe["sector_proxies"], "windows": [20, 60],
            "benchmark": universe.get("benchmark", "SPY"), "sector_records": list(sector_records),
            "issuer_labels_in_observation": sector_labels,
            "return_convention": "log cash-dividend total return; forward-only split continuity",
            "missing_policy": "zero plus mask; require same-session availability and full lookback",
            "classification_policy": "availability AND effective interval; current snapshots never backfilled"}
    if zone_timeframes:
        from rl_stocks.features.market_structure.supply_demand import ZoneConfig
        config = zone_config or ZoneConfig()
        payload["schema"] += "-sd-v1-"+"-".join(zone_timeframes)+"-"+digest(asdict(config))[:12]
        payload["manifest"]["supply_demand"] = {
            "version": "base-impulse-v1", "timeframes": list(zone_timeframes),
            "parameters": asdict(config), "merging": False, "dynamic_thresholds": False,
            "weekly_release": "first observed decision of next week; first week discarded",
            "atr_seed": "first 14 true ranges; Wilder recursion; differs from TA-Lib seed",
            "distance_clip": 20, "strength_log_clip": 10, "age_scale": 252, "age_clip": 10}
    identity = digest(payload)
    destination = Path(output)/identity
    destination.mkdir(parents=True,exist_ok=True)
    target = destination/"dataset.json"
    content=canonical(payload)
    from rl_stocks.core.files import write_once
    write_once(target,content)
    return destination

def read_dataset(path):
    path=Path(path)
    raw=(path/"dataset.json").read_bytes()
    import hashlib
    if hashlib.sha256(raw).hexdigest()!=path.name:
        raise ValueError("dataset checksum mismatch")
    return json.loads(raw)

def normalizer_from(value):
    return Normalizer(tuple(value["names"]),tuple(value["means"]),tuple(value["scales"]))

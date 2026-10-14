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
        return cls(tuple(names), tuple(mean(c) for c in cols), tuple(max(pstdev(c),1e-8) for c in cols))
    def transform(self, values, names):
        if tuple(names)!=self.names or len(values)!=len(self.names):
            raise ValueError("feature schema mismatch")
        if not all(math.isfinite(v) for v in values):
            raise ValueError("nonfinite observation")
        return [(v-m)/s for v,m,s in zip(values,self.means,self.scales)]

def split_name(session):
    return "train" if session <= "2020-12-31" else "validation" if session <= "2023-12-31" else "test"

def generate(snapshot, output, weekly=False, purge=20, allow_unverified=False):
    if purge < 20:
        raise ValueError("purge must cover maximum holding horizon (20 sessions)")
    bars, provenance = load(snapshot)
    if not allow_unverified and (not provenance.get("dividends_verified") or
        any(not x["calendar_verified"] or x["missing"] for x in provenance["coverage"].values())):
        raise ValueError("coverage/corporate actions not verified; audit or explicit research override required")
    episodes, training = {}, []
    names = None
    for symbol in UNIVERSE:
        series = [b for b in bars if b.symbol==symbol]
        if len(series)<80:
            raise ValueError(f"insufficient coverage for {symbol}")
        missing=provenance["coverage"][symbol]["missing"]
        starts=[0]+[i for i in range(1,len(series)) if any(series[i-1].session<day<series[i].session for day in missing)]+[len(series)]
        rows=[]
        for start,end in zip(starts,starts[1:]):
            chunk,names=assemble(series[start:end],weekly)
            rows.extend([{**r,"bar":r["bar"]+start} for r in chunk])
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
                episodes[f"{symbol}-{split}-{k}"] = episode
                if split=="train":
                    training.extend(episode["features"])
    if any(not any(e["split"]==s and e["symbol"]==symbol for e in episodes.values())
           for symbol in UNIVERSE for s in ("train","validation","test")):
        raise ValueError("all symbols must have all three splits")
    normalizer = Normalizer.fit(names,training)
    payload = {"schema":FEATURE_SCHEMA+("-weekly" if weekly else ""), "names":names,
        "normalizer":asdict(normalizer), "episodes":episodes,
        "manifest":{"snapshot":Path(snapshot).name,"provenance":provenance,"universe":UNIVERSE,
            "splits":{"train_end":"2020-12-31","validation_end":"2023-12-31","test_start":"2024-01-01"},
            "purge_sessions":purge,"max_holding_sessions":20,"weekly":weekly,
            "unverified_override":allow_unverified,"survivorship":"Fixed ETF universe chosen retrospectively; no broad-universe claims"}}
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

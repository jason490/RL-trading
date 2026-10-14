from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import os
from rl_stocks.core.contracts import Bar, canonical, digest
from rl_stocks.data.validation.bars import validate

def put(root, bars, provenance, raw_source=None):
    validate(bars)
    payload = {"bars": [asdict(b) for b in bars], "provenance": provenance,
               "raw_source":raw_source}
    identity = digest(payload)
    path = Path(root)/identity
    path.mkdir(parents=True, exist_ok=True)
    content = canonical(payload)
    target = path/"snapshot.json"
    from rl_stocks.core.files import write_once
    write_once(target,content)
    return path

def load(path):
    path = Path(path)
    raw = (path/"snapshot.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != path.name:
        raise ValueError("snapshot checksum mismatch")
    payload = json.loads(raw)
    bars = [Bar(**b) for b in payload["bars"]]
    validate(bars)
    return bars, payload["provenance"]

def export_parquet(path):
    import pandas as pd
    bars, _ = load(path)
    target = Path(path)/"normalized.parquet"
    if target.exists():
        return target
    pd.DataFrame([asdict(b) for b in bars]).to_parquet(target, index=False)
    return target

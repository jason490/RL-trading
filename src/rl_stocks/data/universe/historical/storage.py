"""Immutable, checksummed historical evidence imports."""
import json
import hashlib
from pathlib import Path
from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.files import write_once
from rl_stocks.data.universe.historical.schema import validate


def put_history(payload, output):
    validate(payload)
    target = Path(output)/digest(payload)
    target.mkdir(parents=True, exist_ok=True)
    write_once(target/'history.json', canonical(payload))
    return target


def load_history(path):
    path = Path(path)
    raw = (path/'history.json').read_bytes()
    value = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != path.name:
        raise ValueError('historical evidence checksum mismatch')
    return validate(value)

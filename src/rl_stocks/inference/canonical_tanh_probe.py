"""Fixed128-vector cross-machine probe; no market data or native numeric stack."""

import json
import math
import platform
import sys
from time import perf_counter

from rl_stocks.core.contracts import digest
from rl_stocks.inference.canonical_actor import _fields, _hex, encode32
from rl_stocks.inference.canonical_math import CONTRACT, from_bits32, input32
from rl_stocks.inference.canonical_tanh_policy import CanonicalTanhPolicy

SCHEMA = "canonical-tanh-probe-v1"


def vectors(rows):
    if (
        not isinstance(rows, list)
        or len(rows) != 128
        or any(not isinstance(r, list) or len(r) != 21 for r in rows)
    ):
        raise ValueError("fixed128x21 synthetic vectors required")
    return [[input32(from_bits32(int(_hex(v, 8), 16))) for v in row] for row in rows]


def execute(payload):
    _fields(payload, ("actor", "inputs_hex"))
    actor = CanonicalTanhPolicy(payload["actor"], expected_algorithm="trpo")
    rows = payload["inputs_hex"]
    x = vectors(rows)
    start = perf_counter()
    outputs = [[encode32(v) for v in actor.predict(row)] for row in x]
    elapsed = perf_counter() - start
    return {
        "schema": SCHEMA,
        "arithmetic": CONTRACT,
        "python": platform.python_version(),
        "python_major_minor": list(sys.version_info[:2]),
        "platform": platform.platform(),
        "native_packages_imported": sorted(set(sys.modules) & {"torch", "numpy"}),
        "actor_sha256": payload["actor"]["sha256"],
        "inputs_sha256": digest(rows),
        "points": 128,
        "output_hex": outputs,
        "output_sha256": digest(outputs),
        "elapsed_seconds": elapsed,
        "training_performed": False,
        "market_evaluated": False,
    }


def verify(actual, expected):
    if (
        actual.get("schema") != SCHEMA
        or actual.get("arithmetic") != CONTRACT
        or actual.get("python_major_minor") != [3, 13]
        or actual.get("native_packages_imported") != []
        or actual.get("training_performed") is not False
        or actual.get("market_evaluated") is not False
    ):
        raise ValueError("stdlib default-kernel probe required")
    for key in ("actor_sha256", "inputs_sha256", "points", "output_hex", "output_sha256"):
        if json.dumps(actual.get(key), sort_keys=True) != json.dumps(
            expected.get(key), sort_keys=True
        ):
            raise ValueError("exact tanh reference probe mismatch")
    rows = actual["output_hex"]
    if (
        actual["points"] != 128
        or not isinstance(rows, list)
        or len(rows) != 128
        or any(not isinstance(row, list) or len(row) != 4 for row in rows)
        or digest(rows) != actual["output_sha256"]
    ):
        raise ValueError("complete checksummed128x4 output required")
    for row in rows:
        for value in row:
            if not -1 <= input32(from_bits32(int(_hex(value, 8), 16))) <= 1:
                raise ValueError("bounded finite reference action required")
    elapsed = actual.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 < elapsed < 180:
        raise ValueError("bounded measured reference latency required")
    return True

"""Readiness-matched optional Treasury payloads, without changing original data."""

from copy import deepcopy
import math

from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.data.validation.treasury_coverage import decisions
from rl_stocks.features.registry.assemble import NAMES as BASE_NAMES
from rl_stocks.features.treasury_context import NAMES, context

SCHEMAS = {mode: "daily-indicators-v1-treasury-" + mode + "-v1" for mode in ("masked", "full")}
SOURCE_PROTOCOL_SHA256 = "0beb46986f3279047bd47ac3e245d1a9f3b749c72cae72b9ab3c45e840238161"
AVAILABILITY = "second_following_xnys_close_plus_15m_assumption"


def _hash(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def validate_source(source):
    if (
        source.get("schema") != "treasury-annual-source-v1"
        or source.get("protocol_sha256") != SOURCE_PROTOCOL_SHA256
        or source.get("historical_publication_vintages_verified") is not False
        or len(source.get("receipts", [])) != 15
        or len(set(source["receipts"])) != 15
        or not all(_hash(v) for v in source["receipts"])
    ):
        raise ValueError("accepted bounded Treasury source lineage required")
    return source


def eligible_bounds(ready):
    """Fixed513-bar minimum and20-row preceding purge; no outcome selection."""
    if not ready or any(type(value) is not bool for value in ready):
        raise ValueError("explicit boolean Treasury readiness mask required")
    missing = {i for i, value in enumerate(ready) if not value}
    purged = {j for i in missing for j in range(max(0, i - 20), i)} - missing
    keep = [bool(value and i not in purged) for i, value in enumerate(ready)]
    segments, start = [], None
    for i, value in enumerate(keep + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            segments.append((start, i))
            start = None
    eligible = [s for s in segments if s[1] - s[0] >= 513]
    if len(eligible) != 1:
        raise ValueError(
            "exactly one full-window eligible segment required; no segment/symbol selection"
        )
    shorts = [s for s in segments if s[1] - s[0] < 513]
    first, last = eligible[0]
    counts = dict(
        source_rows=len(ready),
        unavailable=len(missing),
        purged=len(purged),
        short_segment_rows=sum(b - a for a, b in shorts),
        retained=last - first,
    )
    if sum(counts[k] for k in ("unavailable", "purged", "short_segment_rows", "retained")) != len(
        ready
    ):
        raise ValueError("complete Treasury exclusion accounting required")
    return first, last, counts


def build_payload(source, treasury, mode):
    if (
        mode not in SCHEMAS
        or source["schema"] != "daily-indicators-v1"
        or source["names"] != list(BASE_NAMES)
        or source["normalizer"]["names"] != source["names"]
        or set(source["normalizer"]) != {"names", "means", "scales"}
        or "treasury_context" in source["manifest"]
    ):
        raise ValueError("original seven-field source and explicit Treasury mode required")
    validate_source(treasury)
    times, _, _ = decisions(source)
    joined = context(treasury["rows"], sorted(times), decision_times=times)
    Normalizer(
        tuple(source["names"]),
        tuple(source["normalizer"]["means"]),
        tuple(source["normalizer"]["scales"]),
    )
    result = deepcopy(source)
    extra_train, lineage = [], {}
    symbols = set()
    for key, original in source["episodes"].items():
        if set(original) != {"symbol", "split", "bars", "features", "atr"}:
            raise ValueError("plain baseline episode schema required; no unsliced auxiliary fields")
        n = len(original["bars"])
        if n != len(original["features"]) or n != len(original["atr"]):
            raise ValueError("aligned original Treasury consumer rows required")
        if any(len(r) != 7 or not all(math.isfinite(v) for v in r) for r in original["features"]):
            raise ValueError("finite original seven fields required")
        ready = [joined[b["session"]]["ready"] for b in original["bars"]]
        if original["split"] == "train":
            if original["symbol"] in symbols:
                raise ValueError("one original training episode per symbol required")
            symbols.add(original["symbol"])
            first, last, counts = eligible_bounds(ready)
        else:
            if not all(ready):
                raise ValueError("all original validation rows must remain ready; no filtering")
            first, last = 0, n
            counts = dict(source_rows=n, unavailable=0, purged=0, short_segment_rows=0, retained=n)
        episode = result["episodes"][key]
        for field in ("bars", "features", "atr"):
            episode[field] = episode[field][first:last]
        for i, (bar, row) in enumerate(zip(episode["bars"], episode["features"], strict=True)):
            extra = list(joined[bar["session"]]["values"]) if mode == "full" else [0.0, 0.0]
            episode["features"][i] = list(row) + extra
            if mode == "full" and original["split"] == "train":
                extra_train.append(extra)
        lineage[key] = dict(source_bounds=[first, last], **counts)
    fitted = (
        Normalizer.fit(NAMES, extra_train)
        if mode == "full"
        else Normalizer(NAMES, (0.0, 0.0), (1.0, 1.0))
    )
    result["schema"] = SCHEMAS[mode]
    result["names"] = list(BASE_NAMES) + list(NAMES)
    result["normalizer"] = {
        k: source["normalizer"][k] + list(getattr(fitted, k)) for k in ("names", "means", "scales")
    }
    result["manifest"]["treasury_context"] = dict(
        version=1,
        mode=mode,
        names=list(NAMES),
        source_dataset=digest(source),
        source_context=digest(treasury),
        source_protocol_sha256=SOURCE_PROTOCOL_SHA256,
        source_normalizer_sha256=digest(source["normalizer"]),
        source_episodes_sha256=digest(source["episodes"]),
        source_alignment_sha256=digest(joined),
        source_receipts=list(treasury["receipts"]),
        source_snapshot=source["manifest"]["snapshot"],
        eligibility=lineage,
        eligibility_sha256=digest(lineage),
        minimum_training_segment_bars=513,
        purge_before_unready_training_rows=20,
        source_lag_sessions=2,
        maximum_age_xnys_sessions=3,
        maximum_age_calendar_days=7,
        extra_normalizer_fit_rows=len(extra_train),
        normalization_training_only=True,
        availability=AVAILABILITY,
        historical_publication_vintages_verified=False,
        executable=False,
        preserved="Retained equity bars/ATR/original seven fields exact; validation dates/keys unchanged; no test reconstructed",
    )
    return result


def validate_payload_pair(source, payload, treasury, mode):
    if digest(payload) != digest(build_payload(source, treasury, mode)):
        raise ValueError("exact Treasury source/readiness/preprocessing payload required")
    return payload

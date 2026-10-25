"""Immutable opt-in VIX datasets; original equity inputs remain exact."""

from copy import deepcopy
from datetime import date, datetime

from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.data.generation.volatility_context import align_context
from rl_stocks.data.providers.yahoo.volatility import validate_closes
from rl_stocks.features.registry.assemble import NAMES as BASE_NAMES
from rl_stocks.features.volatility_context import NAMES

SCHEMAS = {mode: "daily-indicators-v1-vix-" + mode + "-v1" for mode in ("masked", "full")}
AVAILABILITY = "next_xnys_close_plus_15m_assumption"


def validate_source(source):
    if (
        source.get("symbol") != "^VIX"
        or source.get("provider") != "yahoo"
        or source.get("feed") != "yahoo_index_close"
        or source.get("executable") is not False
        or source.get("historical_publication_vintages") is not False
    ):
        raise ValueError("explicit non-executable current-vintage VIX source required")
    retrieved = datetime.fromisoformat(source["retrieved_time"])
    if retrieved.tzinfo is None or retrieved.utcoffset().total_seconds() != 0:
        raise ValueError("UTC context retrieval provenance required")
    validate_closes(
        source["rows"],
        date.fromisoformat(source["request"]["start"]),
        date.fromisoformat(source["request"]["end_exclusive"]),
    )
    return source


def build_payload(source, volatility, mode):
    if (
        not isinstance(mode, str)
        or mode not in SCHEMAS
        or "vix_context" in source["manifest"]
        or set(source["normalizer"]) != {"names", "means", "scales"}
    ):
        raise ValueError("new explicit masked/full VIX payload required")
    validate_source(volatility)
    joined, counts, _ = align_context(source, volatility["rows"])
    if not all(v["ready"] for v in joined.values()):
        raise ValueError("all original decision rows require exact available VIX context")
    Normalizer(
        tuple(source["normalizer"]["names"]),
        tuple(source["normalizer"]["means"]),
        tuple(source["normalizer"]["scales"]),
    )
    result = deepcopy(source)
    extra_train = []
    for episode in result["episodes"].values():
        for i, (bar, features) in enumerate(zip(episode["bars"], episode["features"], strict=True)):
            extra = list(joined[bar["session"]]["values"]) if mode == "full" else [0.0, 0.0]
            episode["features"][i] = list(features) + extra
            if mode == "full" and episode["split"] == "train":
                extra_train.append(extra)
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
    result["manifest"]["vix_context"] = dict(
        version=1,
        mode=mode,
        symbol="^VIX",
        names=list(NAMES),
        source_dataset=digest(source),
        source_context=digest(volatility),
        source_provider=volatility["provider"],
        source_feed=volatility["feed"],
        source_request={k: volatility["request"][k] for k in ("start", "end_exclusive")},
        retrieved_time=volatility["retrieved_time"],
        source_normalizer_sha256=digest(source["normalizer"]),
        source_episodes_sha256=digest(source["episodes"]),
        source_snapshot=source["manifest"]["snapshot"],
        extra_normalizer_fit_rows=counts["train"] if mode == "full" else 0,
        normalization_training_only=True,
        availability=AVAILABILITY,
        source_lag_sessions=1,
        change_sessions=20,
        historical_publication_vintages_verified=False,
        executable=False,
        preserved="Original seven fields/statistics, equity bars/ATR/row keys/splits exact; no test reconstructed",
    )
    validate_frozen_contract(
        dict(
            feature_schema=result["schema"],
            names=result["names"],
            normalizer=result["normalizer"],
            data_manifest=result["manifest"],
        )
    )
    return result


def validate_payload_pair(source, payload, volatility, mode):
    if digest(payload) != digest(build_payload(source, volatility, mode)):
        raise ValueError("exact immutable VIX source/payload alignment required")
    return payload


def _hash(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def validate_frozen_contract(manifest):
    contract = manifest["data_manifest"].get("vix_context")
    if contract is None:
        if manifest["feature_schema"] in SCHEMAS.values() or set(NAMES) & set(manifest["names"]):
            raise ValueError("VIX schema requires explicit source/availability lineage")
        return None
    mode = contract.get("mode")
    if (
        not isinstance(mode, str)
        or mode not in SCHEMAS
        or type(contract.get("version")) is not int
        or contract["version"] != 1
        or manifest["feature_schema"] != SCHEMAS[mode]
        or contract.get("symbol") != "^VIX"
        or contract.get("source_provider") != "yahoo"
        or contract.get("source_feed") != "yahoo_index_close"
        or contract.get("names") != list(NAMES)
        or manifest["names"] != list(BASE_NAMES) + list(NAMES)
        or set(manifest["normalizer"]) != {"names", "means", "scales"}
        or manifest["normalizer"]["names"] != manifest["names"]
        or contract.get("normalization_training_only") is not True
        or contract.get("historical_publication_vintages_verified") is not False
        or contract.get("executable") is not False
        or contract.get("availability") != AVAILABILITY
        or type(contract.get("source_lag_sessions")) is not int
        or contract["source_lag_sessions"] != 1
        or type(contract.get("change_sessions")) is not int
        or contract["change_sessions"] != 20
        or any(
            not _hash(contract.get(k))
            for k in (
                "source_dataset",
                "source_context",
                "source_normalizer_sha256",
                "source_episodes_sha256",
            )
        )
    ):
        raise ValueError("VIX source/schema/availability contract mismatch")
    requested = contract.get("source_request", {})
    if set(requested) != {"start", "end_exclusive"} or not date.fromisoformat(
        requested["start"]
    ) < date.fromisoformat(requested["end_exclusive"]) <= date(2024, 1, 1):
        raise ValueError("bounded pre2024 VIX source request required")
    retrieved = datetime.fromisoformat(contract["retrieved_time"])
    if retrieved.tzinfo is None or retrieved.utcoffset().total_seconds() != 0:
        raise ValueError("UTC VIX retrieval provenance required")
    Normalizer(
        tuple(manifest["normalizer"]["names"]),
        tuple(manifest["normalizer"]["means"]),
        tuple(manifest["normalizer"]["scales"]),
    )
    if (
        digest({k: manifest["normalizer"][k][:7] for k in ("names", "means", "scales")})
        != contract["source_normalizer_sha256"]
    ):
        raise ValueError("original seven-field preprocessing must remain exact")
    fitted = contract.get("extra_normalizer_fit_rows")
    if type(fitted) is not int or (fitted <= 0 if mode == "full" else fitted != 0):
        raise ValueError("declared training-only VIX preprocessing rows required")
    if mode == "masked" and (
        manifest["normalizer"]["means"][7:] != [0.0, 0.0]
        or manifest["normalizer"]["scales"][7:] != [1.0, 1.0]
    ):
        raise ValueError("masked VIX fields require zero/identity preprocessing")
    return mode

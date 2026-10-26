"""Portable frozen Treasury schema/availability validation; no local cache required."""

from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.data.generation.treasury_payload import AVAILABILITY, SCHEMAS, SOURCE_PROTOCOL_SHA256
from rl_stocks.features.registry.assemble import NAMES as BASE_NAMES
from rl_stocks.features.treasury_context import NAMES


def _hash(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def validate_frozen_contract(manifest):
    contract = manifest["data_manifest"].get("treasury_context")
    if contract is None:
        if manifest["feature_schema"] in SCHEMAS.values() or set(NAMES) & set(manifest["names"]):
            raise ValueError("Treasury schema requires explicit source/availability lineage")
        return None
    if not isinstance(contract, dict):
        raise ValueError("Treasury lineage object required")
    mode = contract.get("mode")
    if (
        not isinstance(mode, str)
        or mode not in SCHEMAS
        or type(contract.get("version")) is not int
        or contract["version"] != 1
        or manifest["feature_schema"] != SCHEMAS[mode]
        or manifest["names"] != list(BASE_NAMES) + list(NAMES)
        or contract.get("names") != list(NAMES)
        or set(manifest["normalizer"]) != {"names", "means", "scales"}
        or manifest["normalizer"]["names"] != manifest["names"]
        or contract.get("source_protocol_sha256") != SOURCE_PROTOCOL_SHA256
        or contract.get("availability") != AVAILABILITY
        or contract.get("normalization_training_only") is not True
        or contract.get("historical_publication_vintages_verified") is not False
        or contract.get("executable") is not False
    ):
        raise ValueError("Treasury frozen schema/availability mismatch")
    for key, value in dict(
        source_lag_sessions=2,
        maximum_age_xnys_sessions=3,
        maximum_age_calendar_days=7,
        minimum_training_segment_bars=513,
        purge_before_unready_training_rows=20,
    ).items():
        if type(contract.get(key)) is not int or contract[key] != value:
            raise ValueError("fixed Treasury lag/age/readiness contract required")
    for key in [
        "source_dataset",
        "source_context",
        "source_normalizer_sha256",
        "source_episodes_sha256",
        "source_alignment_sha256",
        "eligibility_sha256",
    ]:
        if not _hash(contract.get(key)):
            raise ValueError("Treasury source/eligibility identities required")
    receipts = contract.get("source_receipts")
    if (
        not isinstance(receipts, list)
        or len(receipts) != 15
        or not all(_hash(v) for v in receipts)
        or len(set(receipts)) != 15
    ):
        raise ValueError("exact15 Treasury receipt identities required")
    normalizer = manifest["normalizer"]
    Normalizer(tuple(normalizer["names"]), tuple(normalizer["means"]), tuple(normalizer["scales"]))
    if (
        digest({k: normalizer[k][:7] for k in ["names", "means", "scales"]})
        != contract["source_normalizer_sha256"]
    ):
        raise ValueError("original seven Treasury preprocessing statistics changed")
    eligibility = contract.get("eligibility")
    if (
        not isinstance(eligibility, dict)
        or not eligibility
        or digest(eligibility) != contract["eligibility_sha256"]
    ):
        raise ValueError("Treasury complete row-lineage accounting required")
    train_rows = 0
    counts = {"train": 0, "validation": 0}
    for key, row in eligibility.items():
        parts = key.rsplit("-", 2)
        if len(parts) != 3 or parts[1] not in counts or not parts[2].isdigit():
            raise ValueError("source train/development episode key required")
        split = parts[1]
        counts[split] += 1
        fields = ["source_rows", "unavailable", "purged", "short_segment_rows", "retained"]
        if any(type(row.get(k)) is not int or row[k] < 0 for k in fields):
            raise ValueError("integer Treasury exclusion counts required")
        bounds = row.get("source_bounds")
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(type(v) is not int for v in bounds)
            or not 0 <= bounds[0] < bounds[1] <= row["source_rows"]
            or bounds[1] - bounds[0] != row["retained"]
        ):
            raise ValueError("exact retained Treasury row bounds required")
        if (
            sum(row[k] for k in ["unavailable", "purged", "short_segment_rows", "retained"])
            != row["source_rows"]
        ):
            raise ValueError("complete Treasury source/exclusion partition required")
        if split == "train":
            if row["retained"] < 513:
                raise ValueError("full512-transition Treasury training windows required")
            train_rows += row["retained"]
        elif bounds != [0, row["source_rows"]] or any(
            row[k] for k in ["unavailable", "purged", "short_segment_rows"]
        ):
            raise ValueError("all original Treasury validation rows required")
    fitted = contract.get("extra_normalizer_fit_rows")
    if (
        not all(counts.values())
        or type(fitted) is not int
        or fitted != (train_rows if mode == "full" else 0)
    ):
        raise ValueError("exact Treasury training-only fit row count required")
    if mode == "masked" and (
        normalizer["means"][7:] != [0.0, 0.0] or normalizer["scales"][7:] != [1.0, 1.0]
    ):
        raise ValueError("masked Treasury zero/identity preprocessing required")
    return mode

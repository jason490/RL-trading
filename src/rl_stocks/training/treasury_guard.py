"""Payload preprocessing does not authorize an unregistered Treasury training run."""

from rl_stocks.data.generation.treasury_payload import SCHEMAS
from rl_stocks.features.treasury_context import NAMES


def reject_unregistered_treasury(data):
    if (
        data.get("schema") in SCHEMAS.values()
        or "treasury_context" in data.get("manifest", {})
        or set(NAMES) & set(data.get("names", []))
    ):
        raise ValueError(
            "Treasury training requires a separate registered experiment; currently disabled"
        )

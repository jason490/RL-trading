"""Fixed-size projection separating cold startup from work scaling with trees."""

import math


def projection(
    *, cold_constructor_seconds, fit_seconds, acceptance_seconds, warm_constructor_seconds
):
    values = (cold_constructor_seconds, fit_seconds, acceptance_seconds, warm_constructor_seconds)
    if (
        any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)
        or fit_seconds <= 0
    ):
        raise ValueError("finite nonnegative measured intervals and positive fit time required")
    # Cold construction has already completed and must not be charged 768 times.
    # Scale both fitting and parity/export conservatively from2 to256 trees;
    # charge warm construction per model, apply2x margin and reserve900 seconds.
    seconds = (
        2 * (6 * 128 * (fit_seconds + acceptance_seconds) + 6 * warm_constructor_seconds) + 900
    )
    if not math.isfinite(seconds):
        raise ValueError("finite full-operation projection required")
    return {
        "schema": "residual-boosting-component-timing-v1",
        "cold_constructor_seconds": cold_constructor_seconds,
        "prefix_fit_seconds": fit_seconds,
        "prefix_acceptance_seconds": acceptance_seconds,
        "warm_constructor_seconds": warm_constructor_seconds,
        "cold_constructor_already_spent": True,
        "trees_per_prefix": 2,
        "trees_per_model": 256,
        "full_models": 6,
        "safety_multiplier": 2,
        "remaining_work_reserve_seconds": 900,
        "projected_remaining_seconds": seconds,
    }

"""Persistent one-allocation budget for the recurrent SAC learned smoke."""

import json
import math

from rl_stocks.core.contracts import canonical
from rl_stocks.core.files import write_once
from rl_stocks.training.recurrent_sac_smoke import PROTOCOL_SHA256, protocol


def shared_budget(ledger, requested_limit=None, requested_start=None):
    from rl_stocks.runtime.colab.budget import BudgetStop

    runtime = ledger.path.resolve().parent
    if (runtime / "recurrent-sac-smoke-allocation.json").exists():
        raise RuntimeError(
            "Recurrent SAC smoke already claimed its sole allocation; no automatic retry"
        )
    maximum = protocol()["execution"]["maximum_incremental_study_units"]
    limit = maximum if requested_limit is None else requested_limit
    if type(limit) not in (int, float) or not math.isfinite(limit) or not 0 < limit <= maximum:
        raise ValueError("recurrent SAC smoke allowance must be positive and at most0.1 units")
    path = runtime / "recurrent-sac-smoke-budget.json"
    if not path.exists():
        write_once(
            path,
            canonical(
                dict(
                    protocol_sha256=PROTOCOL_SHA256,
                    start_units=ledger.value["consumed"],
                    minimum_account_balance=150,
                )
            ),
        )
    value = json.loads(path.read_text())
    start = value["start_units"]
    if (
        value["protocol_sha256"] != PROTOCOL_SHA256
        or value["minimum_account_balance"] != 150
        or type(start) not in (int, float)
        or not math.isfinite(start)
        or not 0 <= start <= ledger.value["consumed"]
        or requested_start is not None
        and requested_start != start
    ):
        raise ValueError("original recurrent SAC smoke budget baseline must be preserved")
    if ledger.value["consumed"] - start >= limit:
        raise BudgetStop("recurrent SAC smoke allowance exhausted; no automatic retry")
    return float(limit), float(start)


def claim_allocation(ledger, session):
    path = ledger.path.resolve().parent / "recurrent-sac-smoke-allocation.json"
    if path.exists():
        raise RuntimeError("sole recurrent SAC smoke allocation already claimed")
    if not isinstance(session, str) or not session.startswith("rl-stocks-cpu-"):
        raise ValueError("owned CPU diagnostic session required")
    write_once(
        path,
        canonical(
            dict(
                protocol_sha256=PROTOCOL_SHA256,
                session=session,
                allocation_deadline_seconds=900,
                automatic_retry_allowed=False,
            )
        ),
    )

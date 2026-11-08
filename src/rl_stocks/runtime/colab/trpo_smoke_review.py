"""One explicit execution-only review; preserve original engineering study charges."""

import json
from pathlib import Path

from rl_stocks.inference.bundle import checksum
from rl_stocks.runtime.colab.budget import BudgetStop

FAILURE_SHA256 = "f9bfe905969d081d035c1efc412891b935458d41411862167d1345ad2ecb8b05"
FIXED_BOOTSTRAP_SHA256 = "8ce5b7ba50f64c29ca12d0194c50e6b735d64e26cb9eabf4851742479c8c773a"
ORIGINAL_SESSION = "rl-stocks-cpu-4f2bbd67c096"
ORIGINAL_BASE = 40.21369700555672
REVIEW_CLAIM = ".runtime/trpo-native-smoke-bootstrap-review-v1-claim.json"


def review(root, ledger):
    root = Path(root)
    path = root / "results/trpo-smoke-2026-11-08/initial-failure.json"
    if checksum(path) != FAILURE_SHA256:
        raise ValueError("original bootstrap failure receipt required")
    failure = json.loads(path.read_text())
    for name, expected in failure["protected_files"].items():
        if checksum(root / name) != expected:
            raise ValueError("original failed-attempt evidence changed")
    if checksum(root / "scripts/remote_trpo_smoke.py") != FIXED_BOOTSTRAP_SHA256:
        raise ValueError("reviewed kernel-argument bootstrap fix required")
    value = ledger.value
    attempts = value["attempts"]
    if (
        value["budget_stopped"] or not attempts
        or any(a["status"] != "stopped" for a in attempts)
        or attempts[-1]["session"] != ORIGINAL_SESSION
        or attempts[-1]["base_consumed"] != ORIGINAL_BASE
        or attempts[-1].get("training_verified", False)
        or value["consumed"] < failure["project_units"]
        or value["consumed"] >= ORIGINAL_BASE + .25
        or failure["cleanup_verified"] is not True
        or failure["training_performed"] is not False
    ):
        raise BudgetStop("exact stopped pre-fit failure and original study allowance required")
    return {"kind": "reviewed-notebook-argv-fix", "original_session": ORIGINAL_SESSION,
            "original_failure_sha256": FAILURE_SHA256,
            "fixed_bootstrap_sha256": FIXED_BOOTSTRAP_SHA256,
            "original_study_base": ORIGINAL_BASE, "maximum_total_study_units": .25,
            "prior_charges_preserved": value["consumed"] - ORIGINAL_BASE,
            "scientific_protocol_changed": False, "automatic_retry": False}

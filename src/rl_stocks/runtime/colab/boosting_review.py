"""One explicit setup-only continuation; preserves the first claim and charges."""

import hashlib
import json
from pathlib import Path
import re

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.files import write_once

REVIEW_SHA256 = "d5c2985a8107f835a7d7d5f28d3c3b06251c6396dc5a2d2ae38b7d6d6ad88454"
CLAIM = "residual-boosting-reviewed-allocation-v1.json"
COMPONENT_SHA256 = "ecfcec3e90321480db69af12be342f100dfb775385268517eba4b921b856aa1a"
COMPONENT_CLAIM = "residual-boosting-component-allocation-v1.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def spec(root=None, component=False):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[4]
    value = json.loads(
        (
            root
            / (
                "configs/experiments/residual_boosting_component_review.json"
                if component
                else "configs/experiments/residual_boosting_execution_review.json"
            )
        ).read_text()
    )
    if digest(value) != (COMPONENT_SHA256 if component else REVIEW_SHA256):
        raise ValueError("fixed bootstrap execution review required")
    return value


def admission(root, ledger, component=False):
    root = Path(root)
    review = spec(root, component)
    if (ledger.path.parent / (COMPONENT_CLAIM if component else CLAIM)).exists() or (
        ledger.path.parent / "residual-boosting-family-closure-v1.json"
    ).exists():
        raise RuntimeError("review already claimed or predictive family closed")
    if ledger.value["budget_stopped"] or any(
        a["status"] != "stopped" for a in ledger.value["attempts"]
    ):
        raise RuntimeError("original budget stop or unresolved ownership")
    receipt = root / "results/residual-boosting-2026-11-04/bootstrap-failure.json"
    if sha(receipt) != review["failure_receipt_sha256"]:
        raise ValueError("original bootstrap failure changed")
    failure = json.loads(receipt.read_text())
    if (
        failure["worker_started"] is not False
        or failure["model_fitting_performed"] is not False
        or not failure["cleanup_verified"]
    ):
        raise ValueError("setup-only failure and verified cleanup required")
    for name in (
        ".runtime/residual-boosting-fleet-v1.json",
        ".runtime/residual-boosting-allocation-v1.json",
    ):
        if sha(root / name) != failure["protected"][name]:
            raise ValueError("original failed state/claim changed")
    baseline = json.loads((ledger.path.parent / "residual-boosting-budget-v1.json").read_text())
    if (
        baseline["start_units"] != review["original_study_start_units"]
        or baseline["limit_units"] != 0.25
        or baseline["protocol_sha256"] != review["scientific_protocol_sha256"]
    ):
        raise ValueError("original study baseline must not be reset")
    children = [
        c
        for a in ledger.value["attempts"]
        for c in a.get("children", {}).values()
        if c["session"] == review["original_failed_session"]
    ]
    if (
        len(children) != 1
        or children[0]["status"] != "stopped"
        or not children[0].get("cleanup_verified")
    ):
        raise ValueError("original owned shutdown required")
    if ledger.value["consumed"] < review["original_failure_project_units"]:
        raise ValueError("original failure charges missing")
    for name, expected in review.get("required_protected", {}).items():
        if sha(root / name) != expected:
            raise ValueError("prior duration-stop evidence changed")
    for session in review.get("required_stopped_sessions", []):
        owned = [
            c
            for a in ledger.value["attempts"]
            for c in a.get("children", {}).values()
            if c["session"] == session
        ]
        if (
            len(owned) != 1
            or owned[0]["status"] != "stopped"
            or not owned[0].get("cleanup_verified")
        ):
            raise ValueError("prior duration-stop shutdown required")
    for name, expected in review["unchanged_scientific_source_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("scientific source changed: " + name)
    if sha(root / "scripts/install_colab_bootstrap.py") != review["bootstrap_script_sha256"]:
        raise ValueError("bounded bootstrap script changed")
    return 0.25, baseline["start_units"]


def claim(ledger, session, component=False):
    if not re.fullmatch(r"rl-stocks-cpu-[a-f0-9]{12}", session):
        raise ValueError("owned CPU session required")
    path = ledger.path.parent / (COMPONENT_CLAIM if component else CLAIM)
    if path.exists():
        raise RuntimeError("reviewed allocation already claimed; no retry")
    write_once(
        path,
        canonical(
            {
                "session": session,
                "review_sha256": COMPONENT_SHA256 if component else REVIEW_SHA256,
                "automatic_retry": False,
            }
        ),
    )

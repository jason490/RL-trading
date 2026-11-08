import json
from types import SimpleNamespace

import pytest

from rl_stocks.runtime.colab import trpo_smoke_review as review
from rl_stocks.runtime.colab.budget import BudgetStop


def evidence(tmp_path, monkeypatch):
    path = tmp_path / "results/trpo-smoke-2026-11-08/initial-failure.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"protected_files": {"original-claim": "fixture"},
                               "project_units": review.ORIGINAL_BASE + .001,
                               "cleanup_verified": True, "training_performed": False}))
    def checksum(path):
        name = str(path)
        return review.FAILURE_SHA256 if name.endswith("initial-failure.json") else (
            review.FIXED_BOOTSTRAP_SHA256 if name.endswith("remote_trpo_smoke.py") else "fixture")
    monkeypatch.setattr(review, "checksum", checksum)
    value = {"budget_stopped": False, "consumed": review.ORIGINAL_BASE + .001,
             "attempts": [{"status": "stopped", "session": review.ORIGINAL_SESSION,
                           "base_consumed": review.ORIGINAL_BASE}]}
    return SimpleNamespace(value=value)


def test_review_retains_original_claim_and_charges(tmp_path, monkeypatch):
    ledger = evidence(tmp_path, monkeypatch)
    original = json.dumps(ledger.value, sort_keys=True)
    result = review.review(tmp_path, ledger)
    assert result["original_study_base"] == review.ORIGINAL_BASE
    assert result["prior_charges_preserved"] == pytest.approx(.001)
    assert result["maximum_total_study_units"] == .25
    assert not result["automatic_retry"] and not result["scientific_protocol_changed"]
    assert json.dumps(ledger.value, sort_keys=True) == original


@pytest.mark.parametrize("change", ["stop", "active", "session", "base", "budget", "trained"])
def test_review_rejects_wrong_state(tmp_path, monkeypatch, change):
    ledger = evidence(tmp_path, monkeypatch)
    if change == "stop":
        ledger.value["budget_stopped"] = True
    elif change == "active":
        ledger.value["attempts"][-1]["status"] = "active"
    elif change == "session":
        ledger.value["attempts"][-1]["session"] = "other"
    elif change == "base":
        ledger.value["attempts"][-1]["base_consumed"] += .001
    elif change == "budget":
        ledger.value["consumed"] = review.ORIGINAL_BASE + .25
    else:
        ledger.value["attempts"][-1]["training_verified"] = True
    with pytest.raises(BudgetStop):
        review.review(tmp_path, ledger)

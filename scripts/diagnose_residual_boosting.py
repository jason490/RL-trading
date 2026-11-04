"""Fixed saved-forecast audit; no new fitting, correction or gate selection."""

import argparse
import json
from pathlib import Path
import numpy as np

from rl_stocks.core.contracts import canonical
from rl_stocks.core.files import write_once
from rl_stocks.core.provenance import repository_state
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.forecast_errors import decompose
from rl_stocks.reporting.residual_errors import residual_effect

PREDICTIONS = (
    "artifacts/colab/rl-stocks-cpu-c621458a688c/retrieved/artifacts/boosting/probe-predictions.npz"
)
PREDICTIONS_SHA256 = "a3a777b38dd3cec0c5523ffef6e5b59ed9065e4c3bed3af7bb6ec40ce665294c"
ACCEPTANCE = "results/residual-boosting-2026-11-04/acceptance.json"
ACCEPTANCE_SHA256 = "c3535bb9bba19d2745736bda717875efd96035e1777efeeb0533231a4b023115"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    source = repository_state(root)
    if source["dirty"]:
        raise ValueError("clean committed diagnostic source required")
    paths = [
        root / PREDICTIONS,
        root / ACCEPTANCE,
        root / ".runtime/usage-ledger.json",
        root / ".runtime/residual-boosting-family-closure-v1.json",
    ]
    protected = {str(p.relative_to(root)): checksum(p) for p in paths}
    if protected[PREDICTIONS] != PREDICTIONS_SHA256 or protected[ACCEPTANCE] != ACCEPTANCE_SHA256:
        raise ValueError("fixed accepted failed forecast source required")
    acceptance = json.loads((root / ACCEPTANCE).read_text())
    if acceptance["gate"]["passed"] is not False or not acceptance["cleanup_verified"]:
        raise ValueError("closed failed candidate and verified cleanup required")
    records = []
    with np.load(root / PREDICTIONS, allow_pickle=False) as a:
        if len(a["target"]) != 18616 or len(set(a["symbols"])) != 26:
            raise ValueError("complete original probe panel required")
        years = np.array([d[:4] for d in a["sessions"]])
        if set(years) != {"2018", "2019", "2020"}:
            raise ValueError("fixed training-era probe years only")
        for period in ("all", "2018", "2019", "2020"):
            take = np.ones(len(years), dtype=bool) if period == "all" else years == period
            y = a["target"][take]
            days = a["sessions"][take]
            for arm in ("masked", "full"):
                base = a["ridge_" + arm][take]
                for seed in (11, 23, 37):
                    pred = a[f"{arm}-{seed}"][take]
                    records.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "period": period,
                            "forecast_moments": decompose(y, pred, days),
                            "effect_vs_ridge": residual_effect(y, base, pred, days),
                        }
                    )
    assert all(checksum(root / p) == sha for p, sha in protected.items())
    files = [
        "scripts/diagnose_residual_boosting.py",
        "src/rl_stocks/reporting/forecast_errors.py",
        "src/rl_stocks/reporting/residual_errors.py",
    ]
    result = {
        "schema": "residual-boosting-saved-errors-v1",
        "source": source,
        "protected_inputs": protected,
        "source_hashes": {p: checksum(root / p) for p in files},
        "records": records,
        "correction_applied": False,
        "model_loaded": False,
        "training_performed": False,
        "dataset_opened": False,
        "colab_allocated": False,
        "gate_changed": False,
        "development_or_retired_test_access": False,
        "family_reopened": False,
    }
    write_once(args.output, canonical(result))
    print("Recorded24 fixed seed/arm/period diagnostic cells; no fitting or correction.")


if __name__ == "__main__":
    main()

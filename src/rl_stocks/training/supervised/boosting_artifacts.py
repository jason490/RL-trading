"""Independent restoration of completed forecasts; never fits any estimator."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.inference.boosting_signal import BoostingSignalBundle
from rl_stocks.inference.bundle import checksum
from rl_stocks.training.supervised.boosting_spec import protocol, PROTOCOL_SHA256
from rl_stocks.training.supervised.metrics import score, gate
from rl_stocks.training.supervised.forest_spec import TREE_VERSIONS
from rl_stocks.training.supervised.models import design
from rl_stocks.training.supervised.spec import SIGNAL_DATASET, validate_payload


def verify(output, dataset_path):
    output = Path(output)
    protected = {p: checksum(p) for p in output.rglob("*") if p.is_file()}
    data = validate_payload(read_dataset(dataset_path))
    result = json.loads((output / "result.json").read_text())
    if (
        result["protocol_sha256"] != PROTOCOL_SHA256
        or result["fitted_trees"] != 1538
        or result["normalization_unchanged"] is not True
        or any(
            result[k] is not False
            for k in ("project_validation_used", "test_evaluated", "profitability_established")
        )
    ):
        raise ValueError("completed residual-boosting scope mismatch")
    bundle = BoostingSignalBundle(
        output / "frozen" / result["bundle_id"], data["names"], data["source_dataset"]
    )
    m = bundle.manifest
    if (
        m["versions"]["trees"] != TREE_VERSIONS
        or m["protocol"] != protocol()
        or m["signal_dataset"] != SIGNAL_DATASET
        or canonical(m["normalizers"]) != canonical(data["fold_normalizers"])
    ):
        raise ValueError("frozen fitting/normalization contract mismatch")
    fit = [r for r in data["rows"] if r["fold"] == "fit"]
    probe = [r for r in data["rows"] if r["fold"] == "probe"]
    if result["fit_rows"] != len(fit) or result["probe_rows"] != len(probe):
        raise ValueError("complete fit/probe counts required")
    y = np.array([r["target_log_total_return"] for r in fit])
    raw = np.array([r["features"] for r in fit])
    rp = np.array([r["features"] for r in probe])
    expected = {f"{a}-{s}" for a in ("masked", "full") for s in (11, 23, 37)}
    if set(result["portable_max_errors"]) != expected or any(
        not np.isfinite(v) or not 0 <= v <= 1e-10 for v in result["portable_max_errors"].values()
    ):
        raise ValueError("complete native/portable parity required")
    with np.load(output / "probe-predictions.npz", allow_pickle=False) as a:
        if set(a.files) != {
            "target",
            "sessions",
            "symbols",
            "constant",
            "ridge_masked",
            "ridge_full",
            *expected,
        }:
            raise ValueError("complete forecast panel required")
        for key, values in [
            ("target", [r["target_log_total_return"] for r in probe]),
            ("sessions", [r["decision_session"] for r in probe]),
            ("symbols", [r["symbol"] for r in probe]),
        ]:
            np.testing.assert_array_equal(a[key], values)
        np.testing.assert_array_equal(a["constant"], np.full(len(probe), float(y.mean())))
        if m["constant"] != float(y.mean()):
            raise ValueError("fit-fold constant mismatch")
        metrics = {
            "constant": score(a["target"], a["constant"], a["sessions"]),
            "ridge": {},
            "boosting": {},
        }
        frozen_before = canonical({arm: asdict(n) for arm, n in bundle.normalizers.items()})
        for arm in ("masked", "full"):
            x = design(raw, data["fold_normalizers"][arm], arm)
            z = np.column_stack((np.ones(len(x)), x))
            penalty = np.eye(11) * 0.01
            penalty[0, 0] = 0
            coefficients = np.asarray(m["ridge"][arm])
            # Check normal equations directly. Do not refit a model locally.
            np.testing.assert_allclose(
                (z.T @ z / len(z) + penalty) @ coefficients, z.T @ y / len(z), rtol=0, atol=1e-9
            )
            np.testing.assert_allclose(
                bundle.predict(rp, arm, kind="ridge"), a["ridge_" + arm], rtol=0, atol=1e-10
            )
            metrics["ridge"][arm] = score(a["target"], a["ridge_" + arm], a["sessions"])
            metrics["boosting"][arm] = {}
            for seed in (11, 23, 37):
                key = f"{arm}-{seed}"
                pred = bundle.predict(rp, arm, seed)
                np.testing.assert_array_equal(bundle.models[key].ridge, coefficients)
                np.testing.assert_allclose(pred, a[key], rtol=0, atol=1e-10)
                np.testing.assert_array_equal(pred, bundle.predict(rp, arm, seed))
                metrics["boosting"][arm][str(seed)] = score(a["target"], a[key], a["sessions"])
        if canonical({arm: asdict(n) for arm, n in bundle.normalizers.items()}) != frozen_before:
            raise ValueError("inference mutated normalization")
    if digest(metrics) != digest(result["metrics"]) or digest(gate(metrics, "boosting")) != digest(
        result["gate"]
    ):
        raise ValueError("independent predictive metric/gate mismatch")
    if any(checksum(p) != sha for p, sha in protected.items()):
        raise ValueError("artifact mutation during verification")
    return {
        "schema": "residual-boosting-restoration-v1",
        "bundle_id": result["bundle_id"],
        "restored_forecasts": 6 * len(probe),
        "gate": result["gate"],
        "source_files_unchanged": True,
        "local_fitting": False,
        "performance_promotion": False,
    }

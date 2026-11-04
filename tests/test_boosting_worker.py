from types import SimpleNamespace

import numpy as np
import pytest

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.inference.boosting_arrays import save_boosting
from rl_stocks.inference.boosting_signal import BoostingSignalBundle
from rl_stocks.training.supervised.boosting_runner import export, run
from rl_stocks.training.supervised.boosting_spec import protocol
from test_forest_arrays import tree


def test_local_fitting_rejected_before_data_or_output(tmp_path):
    with pytest.raises(RuntimeError, match="Colab"):
        run(tmp_path / "missing", tmp_path / "output", remaining_seconds=7200)
    assert not (tmp_path / "output").exists()


def test_export_checks_actual_geometry_and_contract(tmp_path):
    params = protocol()["model"] | {"n_estimators": 2}
    stub = SimpleNamespace(
        get_params=lambda: params,
        n_features_in_=10,
        estimators_=np.array(
            [[SimpleNamespace(tree_=SimpleNamespace(**tree()))]] * 2, dtype=object
        ),
    )
    export(stub, np.zeros(11), tmp_path / "prefix.npz")
    params["init"] = None
    with pytest.raises(ValueError):
        export(stub, np.zeros(11), tmp_path / "bad.npz")
    assert not (tmp_path / "bad.npz").exists()


def test_bundle_predictions_checksums_and_immutable_normalization(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    weights = {}
    ridge = np.array([0.01] + [0.0] * 10)
    for arm in ("masked", "full"):
        for seed in (11, 23, 37):
            key = f"{arm}-{seed}"
            weights[key] = key + ".npz"
            save_boosting(stage / weights[key], ridge, [tree()] * 256)
    norm = {"names": [str(i) for i in range(10)], "means": [0.0] * 10, "scales": [1.0] * 10}
    manifest = {
        "kind": BoostingSignalBundle.KIND,
        "names": norm["names"],
        "source_dataset": "fixture",
        "weights": weights,
        "ridge": {a: ridge.tolist() for a in ("masked", "full")},
        "constant": 0.01,
        "normalizers": {a: norm for a in ("masked", "full")},
    }
    folder = tmp_path / digest(manifest)
    stage.rename(folder)
    (folder / "manifest.json").write_bytes(canonical(manifest))
    (folder / "checksums.json").write_bytes(
        canonical({p.name: checksum(p) for p in folder.iterdir()})
    )
    model = BoostingSignalBundle(folder, norm["names"], "fixture")
    before = canonical(manifest["normalizers"])
    x = np.zeros((3, 10))
    x[:, 0] = [-1, 0, 1]
    pred = model.predict(x, "full")
    np.testing.assert_allclose(pred, 0.01 + 256 * 0.03 * np.array([-0.02, -0.02, 0.03]), atol=1e-14)
    np.testing.assert_array_equal(pred, model.predict(x, "full"))
    assert canonical(manifest["normalizers"]) == before
    for arm, seed, kind in [
        ("unknown", 11, "boosting"),
        ("full", 99, "boosting"),
        ("full", 11, "wrong"),
    ]:
        with pytest.raises(ValueError):
            model.predict(x, arm, seed, kind)
    for names, source in [(list(reversed(norm["names"])), "fixture"), (norm["names"], "wrong")]:
        with pytest.raises(ValueError):
            BoostingSignalBundle(folder, names, source)
    with pytest.raises(ValueError):
        model.predict(np.full((1, 10), np.nan), "full")
    (folder / "full-11.npz").write_bytes(b"bad")
    with pytest.raises(ValueError):
        BoostingSignalBundle(folder, norm["names"], "fixture")


def test_independent_restoration_rejects_changed_metrics_and_labels(tmp_path, monkeypatch):
    from rl_stocks.training.supervised import boosting_artifacts as verifier
    from rl_stocks.training.supervised.metrics import score, gate
    from rl_stocks.training.supervised.boosting_spec import PROTOCOL_SHA256
    from rl_stocks.training.supervised.spec import SIGNAL_DATASET
    from rl_stocks.training.supervised.forest_spec import TREE_VERSIONS

    fit = [
        {"fold": "fit", "features": [0.0] * 10, "target_log_total_return": 0.01} for _ in range(32)
    ]
    probe = [
        {
            "fold": "probe",
            "features": [0.0] * 10,
            "target_log_total_return": 0.02 + i * 0.0001,
            "decision_session": f"{year}-01-03",
            "symbol": f"S{i}",
        }
        for year in (2018, 2019, 2020)
        for i in range(26)
    ]
    norm = {"names": [str(i) for i in range(10)], "means": [0.0] * 10, "scales": [1.0] * 10}
    data = {
        "rows": fit + probe,
        "names": norm["names"],
        "source_dataset": "fixture",
        "fold_normalizers": {a: norm for a in ("masked", "full")},
    }
    monkeypatch.setattr(verifier, "read_dataset", lambda p: data)
    monkeypatch.setattr(verifier, "validate_payload", lambda v: v)
    stage = tmp_path / "stage"
    stage.mkdir()
    weights = {}
    ridge = np.array([0.01] + [0.0] * 10)
    leaf = {
        "children_left": np.array([-1]),
        "children_right": np.array([-1]),
        "feature": np.array([-2]),
        "threshold": np.array([-2.0]),
        "value": np.array([0.0]),
    }
    for arm in ("masked", "full"):
        for seed in (11, 23, 37):
            key = f"{arm}-{seed}"
            weights[key] = key + ".npz"
            save_boosting(stage / weights[key], ridge, [leaf] * 256)
    manifest = {
        "kind": BoostingSignalBundle.KIND,
        "names": norm["names"],
        "source_dataset": "fixture",
        "signal_dataset": SIGNAL_DATASET,
        "versions": {"trees": TREE_VERSIONS},
        "protocol": protocol(),
        "weights": weights,
        "ridge": {a: ridge.tolist() for a in ("masked", "full")},
        "constant": float(np.mean([r["target_log_total_return"] for r in fit])),
        "normalizers": data["fold_normalizers"],
    }
    bundle = tmp_path / "frozen" / digest(manifest)
    bundle.parent.mkdir()
    stage.rename(bundle)
    (bundle / "manifest.json").write_bytes(canonical(manifest))
    (bundle / "checksums.json").write_bytes(
        canonical({p.name: checksum(p) for p in bundle.iterdir()})
    )
    arrays = {
        "target": np.array([r["target_log_total_return"] for r in probe]),
        "sessions": np.array([r["decision_session"] for r in probe]),
        "symbols": np.array([r["symbol"] for r in probe]),
        **{
            k: np.full(len(probe), 0.01)
            for k in ["constant", "ridge_masked", "ridge_full", *weights]
        },
    }
    metrics = {
        "constant": score(arrays["target"], arrays["constant"], arrays["sessions"]),
        "ridge": {
            a: score(arrays["target"], arrays["ridge_" + a], arrays["sessions"])
            for a in ("masked", "full")
        },
        "boosting": {
            a: {
                str(s): score(arrays["target"], arrays[f"{a}-{s}"], arrays["sessions"])
                for s in (11, 23, 37)
            }
            for a in ("masked", "full")
        },
    }
    result = {
        "protocol_sha256": PROTOCOL_SHA256,
        "fitted_trees": 1538,
        "normalization_unchanged": True,
        "project_validation_used": False,
        "test_evaluated": False,
        "profitability_established": False,
        "bundle_id": bundle.name,
        "fit_rows": len(fit),
        "probe_rows": len(probe),
        "portable_max_errors": dict.fromkeys(weights, 0.0),
        "metrics": metrics,
        "gate": gate(metrics, "boosting"),
    }
    (tmp_path / "result.json").write_bytes(canonical(result))
    np.savez_compressed(tmp_path / "probe-predictions.npz", **arrays)
    assert verifier.verify(tmp_path, "ignored")["restored_forecasts"] == 468
    result["gate"]["passed"] = True
    (tmp_path / "result.json").write_bytes(canonical(result))
    with pytest.raises(ValueError, match="metric/gate"):
        verifier.verify(tmp_path, "ignored")
    result["gate"] = gate(metrics, "boosting")
    (tmp_path / "result.json").write_bytes(canonical(result))
    arrays["target"][0] += 0.1
    np.savez_compressed(tmp_path / "probe-predictions.npz", **arrays)
    with pytest.raises(AssertionError):
        verifier.verify(tmp_path, "ignored")

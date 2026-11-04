"""One fixed Colab-only residual boosting fit; no policy optimization."""

from dataclasses import asdict
import os
from pathlib import Path
import shutil
import time

import numpy as np
from threadpoolctl import threadpool_limits

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.inference.boosting_arrays import save_boosting, load_boosting, predict_boosting
from rl_stocks.inference.boosting_signal import BoostingSignalBundle
from rl_stocks.inference.bundle import checksum
from rl_stocks.inference.forest_arrays import FIELDS
from rl_stocks.training.supervised.boosting_spec import protocol, make_boosting
from rl_stocks.training.supervised.forest_spec import verify_forest_environment
from rl_stocks.training.supervised.models import design, ridge_fit, ridge_predict
from rl_stocks.training.supervised.metrics import score, gate
from rl_stocks.training.supervised.runner import snapshot
from rl_stocks.training.supervised.spec import SIGNAL_DATASET, validate_payload


def export(model, ridge, path):
    params = model.get_params()
    expected = protocol()["model"]
    if (
        any(params[k] != v for k, v in expected.items() if k != "n_estimators")
        or model.n_features_in_ != 10
        or model.estimators_.shape not in ((2, 1), (256, 1))
        or params["n_estimators"] != len(model.estimators_)
    ):
        raise ValueError("fixed fitted residual estimator required")
    trees = [
        {name: np.asarray(getattr(e[0].tree_, name)).reshape(-1) for name in FIELDS}
        for e in model.estimators_
    ]
    save_boosting(path, ridge, trees)


def run(dataset_path, output, *, remaining_seconds):
    from rl_stocks.training.recurrent_sac_learner import require_colab_cpu

    require_colab_cpu()  # Before reading data, creating files or fitting.
    if os.environ.get("RL_COLAB_OWNED_WORKER") != "1":
        raise RuntimeError("residual fitting requires owned Colab worker")
    if not 900 < remaining_seconds <= 7200:
        raise ValueError("bounded remaining allocation time required")
    started = time.monotonic()
    config = protocol()
    versions = verify_forest_environment()
    data = validate_payload(read_dataset(dataset_path))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    stage = output / "models"
    stage.mkdir()
    fit = [r for r in data["rows"] if r["fold"] == "fit"]
    probe = [r for r in data["rows"] if r["fold"] == "probe"]
    raw = np.asarray([r["features"] for r in fit])
    rp = np.asarray([r["features"] for r in probe])
    y = np.asarray([r["target_log_total_return"] for r in fit])
    target = np.asarray([r["target_log_total_return"] for r in probe])
    days = np.asarray([r["decision_session"] for r in probe])
    normalizers = data["fold_normalizers"]
    before = canonical(normalizers)
    x = {a: design(raw, normalizers[a], a) for a in config["arms"]}
    xp = {a: design(rp, normalizers[a], a) for a in x}
    with threadpool_limits(limits=1):
        ridge = {a: ridge_fit(x[a], y).tolist() for a in x}
        residual = {a: y - ridge_predict(ridge[a], x[a]) for a in x}
        stamp = time.monotonic()
        prefix = make_boosting(11, prefix=True)
        cold_seconds = time.monotonic() - stamp
        stamp = time.monotonic()
        prefix.fit(x["full"], residual["full"])
        fit_seconds = time.monotonic() - stamp
        stamp = time.monotonic()
        export(prefix, ridge["full"], output / "prefix.npz")
        native = ridge_predict(ridge["full"], xp["full"]) + prefix.predict(xp["full"])
        np.testing.assert_allclose(
            native,
            predict_boosting(load_boosting(output / "prefix.npz", prefix=True), xp["full"]),
            rtol=0,
            atol=1e-10,
        )
        if not np.isfinite(prefix.train_score_).all():
            raise ValueError("nonfinite prefix loss")
        acceptance_seconds = time.monotonic() - stamp
        stamp = time.monotonic()
        make_boosting(11)  # Unfitted warm constructor only; discarded, no extra fitting.
        warm_seconds = time.monotonic() - stamp
        from rl_stocks.runtime.colab.boosting_timing import projection

        timing = projection(
            cold_constructor_seconds=cold_seconds,
            fit_seconds=fit_seconds,
            acceptance_seconds=acceptance_seconds,
            warm_constructor_seconds=warm_seconds,
        )
        timing["remaining_seconds"] = remaining_seconds - (time.monotonic() - started)
        (output / "timing.json").write_bytes(canonical(timing))
        if timing["projected_remaining_seconds"] >= timing["remaining_seconds"]:
            raise TimeoutError("complete fixed fits/retrieval exceed remaining allocation")
        predictions = {"constant": np.full(len(target), float(y.mean()))}
        metrics = {
            "constant": score(target, predictions["constant"], days),
            "ridge": {},
            "boosting": {a: {} for a in x},
        }
        weights = {}
        summaries = []
        errors = {}
        for arm in config["arms"]:
            predictions["ridge_" + arm] = ridge_predict(ridge[arm], xp[arm])
            metrics["ridge"][arm] = score(target, predictions["ridge_" + arm], days)
            for seed in config["seeds"]:
                if time.monotonic() - started >= remaining_seconds - 900:
                    raise TimeoutError("retrieval/shutdown reserve reached")
                model = make_boosting(seed)
                model.fit(x[arm], residual[arm])
                if not np.isfinite(model.train_score_).all():
                    raise ValueError("nonfinite full loss")
                key = f"{arm}-{seed}"
                weights[key] = key + ".npz"
                export(model, ridge[arm], stage / weights[key])
                predictions[key] = predictions["ridge_" + arm] + model.predict(xp[arm])
                portable = predict_boosting(load_boosting(stage / weights[key]), xp[arm])
                np.testing.assert_allclose(predictions[key], portable, rtol=0, atol=1e-10)
                errors[key] = float(np.max(np.abs(predictions[key] - portable)))
                metrics["boosting"][arm][str(seed)] = score(target, predictions[key], days)
                summaries.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "trees": 256,
                        "fit_loss_curve": model.train_score_.tolist(),
                    }
                )
                (output / "progress.json").write_bytes(canonical(summaries))
                snapshot(output)
    assert canonical(normalizers) == before
    manifest = {
        "kind": BoostingSignalBundle.KIND,
        "protocol": config,
        "signal_dataset": SIGNAL_DATASET,
        "source_dataset": data["source_dataset"],
        "names": data["names"],
        "normalizers": normalizers,
        "weights": weights,
        "ridge": ridge,
        "constant": float(y.mean()),
        "versions": versions,
    }
    bundle = output / "frozen" / digest(manifest)
    bundle.mkdir(parents=True)
    for name in weights.values():
        shutil.copyfile(stage / name, bundle / name)
    (bundle / "manifest.json").write_bytes(canonical(manifest))
    (bundle / "checksums.json").write_bytes(
        canonical({p.name: checksum(p) for p in bundle.iterdir()})
    )
    frozen = BoostingSignalBundle(bundle, data["names"], data["source_dataset"])
    frozen_before = canonical({a: asdict(n) for a, n in frozen.normalizers.items()})
    for arm in x:
        for seed in config["seeds"]:
            pred = frozen.predict(rp, arm, seed)
            np.testing.assert_allclose(pred, predictions[f"{arm}-{seed}"], rtol=0, atol=1e-10)
            np.testing.assert_array_equal(pred, frozen.predict(rp, arm, seed))
    assert frozen_before == canonical({a: asdict(n) for a, n in frozen.normalizers.items()})
    result = {
        "protocol_sha256": digest(config),
        "bundle_id": bundle.name,
        "metrics": metrics,
        "gate": gate(metrics, "boosting"),
        "portable_max_errors": errors,
        "fit_rows": len(fit),
        "probe_rows": len(probe),
        "fitted_trees": 1538,
        "normalization_unchanged": True,
        "project_validation_used": False,
        "test_evaluated": False,
        "profitability_established": False,
    }
    np.savez_compressed(
        output / "probe-predictions.npz",
        target=target,
        sessions=days,
        symbols=np.asarray([r["symbol"] for r in probe]),
        **predictions,
    )
    (output / "result.json").write_bytes(canonical(result))
    snapshot(output)
    return output, bundle

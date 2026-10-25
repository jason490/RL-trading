"""Read-only restoration of owned VIX recurrent checkpoints and frozen sequences."""

from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.inference.bundle import FrozenPolicy, checksum
from rl_stocks.rl.registry.factory import CLASSES
from rl_stocks.runtime.colab.distributional_artifacts import DEVELOPMENT_SYMBOLS
from rl_stocks.training.recurrent_restore import (
    same_restored as same_restored,
    check_prediction as check_prediction,
)
from rl_stocks.training.vix_context import (
    PROTOCOL_SHA256,
    BINDING_SHA256,
    binding,
    model_record,
    validate_record,
    validate_payload,
)


def verify_owned_run(run_path, bundle_path, data, arm, smoke=False, seed=11):
    """First validate owned archive/source; never deserialize unrelated model files."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    with torch.random.fork_rng(devices=[]):
        try:
            return _verify(Path(run_path), Path(bundle_path), data, arm, smoke, seed)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def _verify(run_path, bundle_path, data, arm, smoke, seed):
    validate_payload(data, arm)
    identity = binding()["datasets"][arm]
    m = json.loads((run_path / "metadata.json").read_text())
    if (
        m["status"] != "complete"
        or m["vix_arm"] != arm
        or m["smoke"] is not smoke
        or m["seed"] != seed
        or m["algorithm"] != "recurrent_ppo"
        or m["dataset"] != identity
        or m["study_protocol_sha256"] != PROTOCOL_SHA256
        or m["vix_binding_sha256"] != BINDING_SHA256
    ):
        raise ValueError("owned completed VIX run identity mismatch")
    paths = [
        run_path / name
        for name in ("metadata.json", "best.zip", "latest.zip", "vix-initialization.npz")
    ]
    paths += [p for p in bundle_path.iterdir() if p.is_file()]
    protected = {str(p): checksum(p) for p in paths}
    models = {}
    for name, filename, field in [
        ("latest", "latest.zip", "vix_final"),
        ("selected", "best.zip", "vix_selected"),
    ]:
        candidate = CLASSES["recurrent_ppo"].load(run_path / filename, device="cpu")
        record = model_record(candidate)
        validate_record(
            record,
            arm,
            smoke,
            selected_steps=m["selected_step"] if name == "selected" else None,
            seed=seed,
        )
        if not same_restored(record, m[field]):
            raise ValueError("actual saved VIX weights/Adam/provenance did not restore")
        models[name] = candidate
    manifest = json.loads((bundle_path / "manifest.json").read_text())
    feed = data["manifest"]["provenance"]["capabilities"]["feed"]
    frozen = FrozenPolicy(bundle_path, data["schema"], feed, "cpu")
    if (
        manifest["names"] != data["names"]
        or manifest["normalizer"] != data["normalizer"]
        or manifest["data_manifest"] != data["manifest"]
        or checksum(run_path / "best.zip") != checksum(bundle_path / "model.zip")
        or not same_restored(model_record(frozen._model), m["vix_selected"])
    ):
        raise ValueError("complete frozen VIX data/model/Adam state mismatch")
    normalization = canonical(asdict(frozen.normalizer))
    symbols, predictions = [], 0
    first_observation = None
    for episode in data["episodes"].values():
        if episode["split"] != "validation":
            continue
        env = TradingEnv(
            [episode],
            normalizer_from(data["normalizer"]),
            data["names"],
            SimulationConfig(**m["simulation"]),
            observation_profile="execution_masked",
            exposure_execution="unit_magnitude",
        )
        try:
            observation, _ = env.reset(seed=seed, options={"episode": 0})
            first_observation = observation.copy()
            left = right = None
            first_action = first_state = None
            for index in range(3):
                action, left, right = check_prediction(
                    models["selected"], frozen, observation, left, right, index == 0
                )
                if index == 0:
                    first_action, first_state = action.copy(), tuple(v.copy() for v in right)
                predictions += 1
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            reset_action, reset_state = frozen.predict_recurrent(
                first_observation, state=None, episode_start=True
            )
            np.testing.assert_array_equal(reset_action, first_action)
            for a, b in zip(reset_state, first_state, strict=True):
                np.testing.assert_array_equal(a, b)
        finally:
            env.close()
        symbols.append(episode["symbol"])
    if len(symbols) != 26 or set(symbols) != set(DEVELOPMENT_SYMBOLS) or predictions < 78:
        raise ValueError("all26 VIX development prefixes and at least78 predictions required")
    for schema, source in [("incompatible-schema", feed), (data["schema"], "incompatible-feed")]:
        try:
            FrozenPolicy(bundle_path, schema, source, "cpu")
        except ValueError:
            pass
        else:
            raise ValueError("incompatible frozen VIX request accepted")
    probes = [np.full(frozen.observation_size, np.nan, dtype=np.float32)]
    if arm == "masked":
        invalid = first_observation.copy()
        invalid[7:9] = 1.0
        probes.append(invalid)
    for invalid in probes:
        try:
            frozen.predict_recurrent(invalid, state=None, episode_start=True)
        except ValueError:
            pass
        else:
            raise ValueError("invalid or nonzero masked VIX observation accepted")
    if (
        normalization != canonical(asdict(frozen.normalizer))
        or not same_restored(model_record(frozen._model), m["vix_selected"])
        or not same_restored(model_record(models["latest"]), m["vix_final"])
        or not same_restored(model_record(models["selected"]), m["vix_selected"])
        or digest(data) != identity
        or any(checksum(p) != sha for p, sha in protected.items())
    ):
        raise ValueError("VIX verification changed model/optimizer/normalization/data/files")
    return dict(
        actual_saved_models_restored=True,
        latest_selected_and_frozen_optimizers_exact=True,
        frozen_predictions=predictions,
        symbols_checked=symbols,
        recurrent_sequence_and_reset_parity=True,
        caller_state_unchanged=True,
        schema_feed_nonfinite_rejection=True,
        masked_observation_rejection=arm == "masked",
        normalization_unchanged=True,
        protected_source_sha256=protected,
        source_files_unchanged=True,
        optimizer_steps=0,
    )

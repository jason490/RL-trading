"""Colab-only finite-update smoke/profile; never performance selection."""

import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from sb3_contrib import TRPO

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.provenance import repository_state, source_identity
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.evaluation.offpolicy_fit import fingerprint
from rl_stocks.inference.bundle import checksum
from rl_stocks.inference.canonical_actor import encode32
from rl_stocks.inference.canonical_tanh_export import export_policy
from rl_stocks.inference.canonical_tanh_policy import CanonicalTanhPolicy
from rl_stocks.training.checkpoint_export import export_artifacts
from rl_stocks.training.recurrent_sac_learner import require_colab_cpu
from rl_stocks.training.sac_update_worker import preserved_rng
from rl_stocks.training.task_backend import task_execution_backend
from rl_stocks.training.trpo_audit import UpdateAudit
from rl_stocks.training.trpo_feasibility import create, initial_parity

PROTOCOL_SHA256 = "e0b8a26cbfd51eee4046544d49e282abd1409b024ec4a0637d92b774bffc5f7b"


def protocol():
    path = Path(__file__).resolve().parents[3] / "configs/experiments/trpo_smoke.json"
    value = json.loads(path.read_text())
    if digest(value) != PROTOCOL_SHA256:
        raise ValueError("fixed TRPO smoke protocol required")
    return value


def validate_dataset(dataset):
    spec = protocol()
    if (
        digest(dataset) != spec["dataset"]
        or digest(dataset["normalizer"]) != spec["normalizer_sha256"]
        or len(dataset["names"]) != 7
        or dataset["manifest"]["included_splits"] != ["train", "validation"]
        or any(e["split"] not in ("train", "validation") for e in dataset["episodes"].values())
    ):
        raise ValueError("fixed causal train/development payload required; retired data rejected")
    return dataset


def build_environment(dataset, *, smoke):
    if type(smoke) is not bool:
        raise ValueError("explicit smoke mode required")
    validate_dataset(dataset)
    return TradingEnv(
        [e for e in dataset["episodes"].values() if e["split"] == "train"],
        normalizer_from(dataset["normalizer"]),
        dataset["names"],
        random_start=True,
        episode_steps=128 if smoke else 512,
        start_sampling="stock_etf_cycle_windows",
        observation_profile="execution_full",
        reward_kind="net_log_equity",
        reward_scale=1.0,
        exposure_execution="unrestricted",
    )


def run(dataset, output):
    require_colab_cpu()  # Before path/data construction or any learning.
    validate_dataset(dataset)
    spec = protocol()
    output = Path(output)
    if output.exists() or output.parent.name != "artifacts":
        raise ValueError("fresh owned smoke output required")
    git = repository_state(Path.cwd())
    if git["dirty"]:
        raise ValueError("committed smoke source required")
    output.mkdir(parents=True)
    normalizer_before = canonical(dataset["normalizer"])
    results = []
    with preserved_rng():
        torch.set_num_threads(1)
        for mode in ("smoke", "profile"):
            envs = {}
            try:
                for a in ("ppo", "trpo"):
                    envs[a] = build_environment(dataset, smoke=mode == "smoke")
                models = {a: create(a, envs[a], seed=11, smoke=mode == "smoke") for a in envs}
                parity = initial_parity(models["ppo"], models["trpo"])
                for algorithm in ("ppo", "trpo"):
                    case = next(
                        c
                        for c in spec["cases"]
                        if c["mode"] == mode and c["algorithm"] == algorithm
                    )
                    folder = output / f"{mode}-{algorithm}"
                    folder.mkdir()
                    model = models[algorithm]
                    model.set_random_seed(11)
                    model.save(folder / "initial.zip")
                    export_policy(
                        folder / "initial.zip",
                        folder / "reference-initial.json",
                        algorithm=algorithm,
                    )
                    export_artifacts(output.parent)
                    audit = UpdateAudit(
                        model,
                        algorithm,
                        expected_updates=case["nominal_updates"],
                        expected_adam_steps=case["adam_steps"],
                    )
                    try:
                        started = time.perf_counter()
                        model.learn(total_timesteps=case["steps"])
                        elapsed = time.perf_counter() - started
                        proof = audit.verify(case["steps"])
                        states = audit.actor_states
                    finally:
                        audit.close()
                    if "train" in model.__dict__ or "hessian_vector_product" in model.__dict__:
                        raise ValueError("live observers cannot enter native checkpoint")
                    model.save(folder / "final.zip")
                    torch.save(states, folder / "actor-updates.pt")
                    export_policy(
                        folder / "final.zip", folder / "reference-final.json", algorithm=algorithm
                    )
                    original = fingerprint(
                        (model.policy.state_dict(), model.policy.optimizer.state_dict())
                    )
                    observations = states[-1]["observations"]
                    indices = np.linspace(0, len(observations) - 1, 8, dtype=int)
                    x = observations[indices].numpy().astype(np.float32)
                    with preserved_rng(), torch.no_grad():
                        frozen = (PPO if algorithm == "ppo" else TRPO).load(
                            folder / "final.zip", device="cpu"
                        )
                        native = model.predict(x, deterministic=True)[0]
                        np.testing.assert_array_equal(
                            native, frozen.predict(x, deterministic=True)[0]
                        )
                        np.testing.assert_array_equal(
                            native, model.predict(x, deterministic=True)[0]
                        )
                    if original != fingerprint(
                        (model.policy.state_dict(), model.policy.optimizer.state_dict())
                    ):
                        raise ValueError("native inference mutated learned state")
                    reference = CanonicalTanhPolicy.load(
                        folder / "reference-final.json",
                        expected_source=checksum(folder / "final.zip"),
                        expected_algorithm=algorithm,
                    )
                    output_bits = [
                        [encode32(v) for v in reference.predict(row.tolist())] for row in x
                    ]
                    if normalizer_before != canonical(dataset["normalizer"]):
                        raise ValueError("normalizer mutated")
                    record = {
                        "case": case,
                        "initial_parity": parity,
                        "learn_seconds": elapsed,
                        "audit": proof,
                        "actor_updates_sha256": checksum(folder / "actor-updates.pt"),
                        "initial_model_sha256": checksum(folder / "initial.zip"),
                        "final_model_sha256": checksum(folder / "final.zip"),
                        "reference_initial_sha256": checksum(folder / "reference-initial.json"),
                        "reference_final_sha256": checksum(folder / "reference-final.json"),
                        "native_save_load_exact": True,
                        "normalization_unchanged": True,
                        "reference_inputs_hex": [[encode32(v) for v in row] for row in x],
                        "reference_output_hex": output_bits,
                        "sampling": envs[algorithm].sampling_statistics(),
                        "backend": task_execution_backend(model.device),
                        "performance_evaluated": False,
                    }
                    (folder / "result.json").write_bytes(canonical(record))
                    results.append(record)
                    (output / "progress.json").write_bytes(
                        canonical({"completed_cases": len(results), "results": results})
                    )
                    export_artifacts(output.parent)
            finally:
                for env in envs.values():
                    env.close()
    validate_dataset(dataset)  # Training must not mutate the immutable source payload.
    result = {
        "schema": "trpo-smoke-result-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "dataset": spec["dataset"],
        "normalizer_sha256": spec["normalizer_sha256"],
        "source": git,
        "source_hash": source_identity(),
        "normalizer": dataset["normalizer"],
        "names": dataset["names"],
        "results": results,
        "training_location": "Colab CPU",
        "development_evaluated": False,
        "retired_test_access": False,
        "performance_promotion": False,
    }
    (output / "result.json").write_bytes(canonical(result))
    export_artifacts(output.parent)
    return result

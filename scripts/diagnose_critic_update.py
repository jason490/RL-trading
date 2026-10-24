"""Bounded inference-only diagnostic; run with the original source PYTHONPATH."""

import argparse
import importlib.util
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.files import write_once
from rl_stocks.core.provenance import source_identity
from rl_stocks.inference.bundle import checksum
from rl_stocks.rl.registry.factory import CLASSES
from rl_stocks.training.distributional_critics import capture_global_rng, restore_global_rng
from rl_stocks.training.distributional_records import model_record, validate_record
from rl_stocks.training.distributional_restoration import _same_restored

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "54efbcc2e898c8323e2d43ad0d57767feb1d2fc57d72fc2284bf65a3fdafd99c"


def save_journal(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(output):
    started = time.monotonic()
    config_path = ROOT / "configs/experiments/critic_update_diagnostic.json"
    spec = json.loads(config_path.read_text())
    if digest(spec) != PROTOCOL_SHA256 or source_identity() != spec["source_hash"]:
        raise ValueError("fixed diagnostic and original frozen package required")
    if output.exists():
        raise ValueError("new diagnostic output required")
    summary_path, cleanup_path = (ROOT / spec[k] for k in ("source_summary", "cleanup"))
    helper_path = ROOT / spec["helper"]
    for path, expected in (
        (summary_path, spec["source_summary_sha256"]),
        (cleanup_path, spec["cleanup_sha256"]),
        (helper_path, spec["helper_sha256"]),
    ):
        if checksum(path) != expected:
            raise ValueError("fixed accepted source/helper checksum mismatch")
    helper_spec = importlib.util.spec_from_file_location("critic_update_numeric", helper_path)
    numeric = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(numeric)
    summary = json.loads(summary_path.read_text())
    cleanup = json.loads(cleanup_path.read_text())
    state_path = ROOT / spec["state"]
    state = json.loads(state_path.read_text())
    if (
        state["status"] != "complete"
        or state.get("cleanup_verified") is not True
        or summary["screen"]["passed"] is not False
        or summary["test_evaluated"] is not False
        or cleanup["retained_files_unchanged"] is not True
    ):
        raise ValueError("accepted failed owned-cleaned source study required")
    protected = {
        str(p): checksum(p)
        for p in (config_path, summary_path, cleanup_path, helper_path, state_path, Path(__file__))
    }
    runs, models = {}, {}
    for arm in spec["arms"]:
        item = state["completed"][f"critic-{arm}-11"]
        run_path = Path(item["root"]) / "runs" / summary["cells"][arm]["run"]
        metadata_path = run_path / "metadata.json"
        expected = summary["cells"][arm]["restoration"]["protected_source_sha256"][
            str(metadata_path)
        ]
        if checksum(metadata_path) != expected:
            raise ValueError("accepted critic metadata mismatch")
        protected[str(metadata_path)] = expected
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata["source_hash"] != spec["source_hash"]
            or metadata["actual_steps"] != spec["transitions"]
        ):
            raise ValueError("fixed actual50k source required")
        pair = {}
        for label, filename in (("before", spec["before_file"]), ("after", spec["after_file"])):
            path = run_path / filename
            if checksum(path) != cleanup["retained_files_sha256"][str(path)]:
                raise ValueError("retained accepted checkpoint bytes mismatch")
            protected[str(path)] = checksum(path)
            model = CLASSES[metadata["algorithm"]].load(path, device="cpu")
            model.policy.set_training_mode(False)
            record = model_record(model)
            validate_record(
                record, arm, selected_steps=spec["transitions"] if label == "before" else None
            )
            if record["updates"] != spec[label + "_updates"]:
                raise ValueError("exact one-update checkpoint pair required")
            if label == "after" and not _same_restored(record, metadata["critic_final"]):
                raise ValueError("actual accepted final optimizer/weights mismatch")
            pair[label] = model
        models[arm] = pair
        runs[arm] = run_path
    reference = models["tqc_truncated"]["after"]
    replay_path = runs["tqc_truncated"] / "replay.pkl"
    if checksum(replay_path) != cleanup["retained_files_sha256"][str(replay_path)]:
        raise ValueError("accepted candidate training replay required")
    protected[str(replay_path)] = checksum(replay_path)
    prior = model_record(reference)
    reference.load_replay_buffer(replay_path)
    if not _same_restored(model_record(reference), prior):
        raise ValueError("replay loading changed model")
    buffer = reference.replay_buffer
    if buffer.pos != spec["transitions"] or buffer.full or buffer.observations.shape[1:] != (1, 21):
        raise ValueError("exact accepted replay observation contract required")
    indices = np.linspace(0, spec["transitions"] - 1, spec["observation_count"], dtype=int)
    observations = buffer.observations[indices, 0].copy()
    if not np.isfinite(observations).all():
        raise ValueError("finite fixed training observations required")
    records = {}
    before_records = {
        a: {p: model_record(m) for p, m in pair.items()} for a, pair in models.items()
    }
    for arm, pair in models.items():
        actions, parameters = {}, {}
        with torch.no_grad():
            for phase, model in pair.items():
                actions[phase] = model.predict(observations, deterministic=True)[0]
                mean, log_std, _ = model.actor.get_action_dist_params(torch.as_tensor(observations))
                parameters[phase] = (mean.numpy(), log_std.numpy())

        def tensors(module):
            return {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}

        record = dict(
            actor=numeric.tensor_change(
                tensors(pair["before"].actor), tensors(pair["after"].actor)
            ),
            critic=numeric.tensor_change(
                tensors(pair["before"].critic), tensors(pair["after"].critic)
            ),
            actions=numeric.action_change(actions["before"], actions["after"]),
            gaussian_kl=numeric.normal_kl(*parameters["before"], *parameters["after"]),
        )
        for phase, model in pair.items():
            if canonical(model_record(model)) != canonical(before_records[arm][phase]):
                raise ValueError("inference changed weights, optimizer or RNG")
        records[arm] = record
        save_journal(
            ROOT / ".runtime/critic-update-progress.json", dict(status="running", completed=records)
        )
    if any(checksum(Path(path)) != expected for path, expected in protected.items()):
        raise ValueError("diagnostic changed protected files")
    result = dict(
        protocol_sha256=PROTOCOL_SHA256,
        records=records,
        observation_indices=indices.tolist(),
        observations_sha256=digest(observations.tolist()),
        source_files_sha256=protected,
        source_files_unchanged=True,
        weights_optimizer_unchanged=True,
        duration_seconds=time.monotonic() - started,
        optimizer_steps=0,
        training_performed=False,
        returns_recomputed=False,
        test_evaluated=False,
        checkpoint_selected=False,
        colab_allocated=False,
        interpretation=spec["interpretation"],
    )
    write_once(output, (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    progress = ROOT / ".runtime/critic-update-progress.json"
    if progress.exists():
        raise ValueError("existing diagnostic journal requires review; no automatic restart")
    write_once(progress, b'{"status":"starting"}\n')

    def expired(*_):
        raise TimeoutError("fixed120-second diagnostic deadline")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(120)
    rng = capture_global_rng()
    torch_rng = torch.get_rng_state().clone()
    try:
        torch.set_num_threads(1)
        result = run(args.output)
        save_journal(progress, dict(status="complete", duration_seconds=result["duration_seconds"]))
    except BaseException as error:
        save_journal(progress, dict(status="failed", error_type=type(error).__name__))
        raise
    finally:
        restore_global_rng(rng)
        torch.set_rng_state(torch_rng)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    print(
        "Compared three accepted one-update pairs on256 fixed training observations; no optimization or return scoring."
    )


if __name__ == "__main__":
    main()

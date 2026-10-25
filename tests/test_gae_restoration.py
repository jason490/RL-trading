"""Real recurrent/Adam serialization and simulation fixtures; no optimization."""

from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from test_gae_artifacts import archive
from test_gae_initialization import model as construct
from rl_stocks.core.contracts import Bar, digest
from rl_stocks.inference.bundle import freeze
from rl_stocks.training import gae_initialization as study, gae_restoration as restore
from rl_stocks.training.gae_spec import protocol
from rl_stocks.training.vix_initialization import seeded_fingerprints


def fixture(root, monkeypatch, arm):
    archive(root, arm)
    run = root / "artifacts/runs/fixture"
    metadata = json.loads((run / "metadata.json").read_text())
    old = root / "artifacts/frozen" / metadata.pop("bundle_id")
    manifest = json.loads((old / "manifest.json").read_text())
    model = construct(arm)
    study.apply_initialization(model, arm, smoke=True)
    model.num_timesteps, model._n_updates = 512, 40
    for parameter in model.policy.parameters():
        model.policy.optimizer.state[parameter] = dict(
            step=torch.tensor(80.0),
            exp_avg=torch.full_like(parameter, 0.01),
            exp_avg_sq=torch.full_like(parameter, 0.02),
        )
    with torch.no_grad():
        next(model.policy.parameters()).add_(0.001)
    final = study.model_record(model)
    metadata.update(gae_final=deepcopy(final), gae_selected=deepcopy(final))
    model.save(run / "best.zip")
    model.save(run / "latest.zip")
    data = dict(
        schema=manifest["feature_schema"],
        names=manifest["names"],
        normalizer=manifest["normalizer"],
        manifest=manifest["data_manifest"],
        episodes={},
    )
    for symbol in restore.DEVELOPMENT_SYMBOLS:
        bars = []
        for index in range(4):
            day = f"2021-01-{index + 4:02d}"
            bars.append(
                asdict(
                    Bar(
                        symbol=symbol,
                        session=day,
                        open_time=day + "T14:30:00+00:00",
                        event_time=day + "T21:00:00+00:00",
                        available_time=day + "T21:15:00+00:00",
                        retrieved_time="2026-10-25T00:00:00+00:00",
                        provider="fixture",
                        feed="fixture",
                        revision="fixture",
                        open=100.0,
                        high=101.0,
                        low=99.0,
                        close=100.0,
                        volume=1000.0,
                    )
                )
            )
        features = [[0.1] * len(data["names"]) for _ in bars]
        data["episodes"][symbol] = dict(
            symbol=symbol, split="validation", bars=bars, features=features, atr=[1.0] * len(bars)
        )
    data["manifest"]["provenance"] = dict(capabilities=dict(feed="fixture"))
    identity = digest(data)
    metadata["dataset"] = identity
    bundle = freeze(run / "best.zip", data, metadata, root / "artifacts/frozen")
    metadata["bundle_id"] = bundle.name
    (run / "metadata.json").write_text(json.dumps(metadata))
    # Synthetic fixture replaces only dataset binding; real model/protocol/start
    # validators and the actual frozen loader remain active.
    bound = protocol()
    bound["dataset"] = identity
    monkeypatch.setattr(restore, "protocol", lambda: bound)
    monkeypatch.setattr(restore, "validate_payload", lambda d: d)
    return run, bundle, data


@pytest.mark.parametrize("arm", ["control", "mc_trace"])
def test_actual_checkpoints_adam_frozen_sequences_and_caller_rng_restore(
    tmp_path, monkeypatch, arm
):
    run, bundle, data = fixture(tmp_path, monkeypatch, arm)
    before = deepcopy(data)
    rng = seeded_fingerprints()
    torch_rng = torch.get_rng_state().clone()
    result = restore.verify_owned_run(run, bundle, data, arm, smoke=True)
    assert result["actual_saved_models_restored"] and result["frozen_predictions"] == 78
    assert result["latest_selected_and_frozen_optimizers_exact"] and result["optimizer_steps"] == 0
    assert result["actual_rollout_arrays_verified"] and result["rollouts"] == 4
    assert seeded_fingerprints() == rng and torch.equal(torch.get_rng_state(), torch_rng)
    assert data == before


def test_recorded_moments_must_match_actual_saved_model_and_rng_restores_on_failure(
    tmp_path, monkeypatch
):
    run, bundle, data = fixture(tmp_path, monkeypatch, "mc_trace")
    path = run / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["gae_final"]["optimizer_sha256"] = "0" * 64
    path.write_text(json.dumps(metadata))
    rng = seeded_fingerprints()
    torch_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="did not restore"):
        restore.verify_owned_run(run, bundle, data, "mc_trace", smoke=True)
    assert seeded_fingerprints() == rng and torch.equal(torch.get_rng_state(), torch_rng)


def test_sequence_verifier_rejects_lost_memory_or_mutated_caller_state():
    class Model:
        def predict(self, observation, *, state, episode_start, deterministic):
            old = np.zeros((1, 1, 1)) if episode_start[0] else state[0]
            new = old + float(observation[0])
            return np.full(4, np.tanh(float(new.item()))), (new, new.copy())

    class Adapter:
        exposure_execution = "unrestricted"

        def predict_recurrent(self, observation, *, state, episode_start):
            return Model().predict(
                observation,
                state=state,
                episode_start=np.array([episode_start]),
                deterministic=True,
            )

    _, left, right = restore.check_prediction(Model(), Adapter(), np.array([1.0]), None, None, True)

    class Forgetting(Adapter):
        def predict_recurrent(self, observation, *, state, episode_start):
            return super().predict_recurrent(observation, state=None, episode_start=True)

    with pytest.raises(AssertionError):
        restore.check_prediction(Model(), Forgetting(), np.array([1.0]), left, right, False)

    class Mutating(Adapter):
        def predict_recurrent(self, observation, *, state, episode_start):
            if state is not None:
                state[0][...] += 1
            return super().predict_recurrent(observation, state=state, episode_start=episode_start)

    with pytest.raises(AssertionError):
        restore.check_prediction(Model(), Mutating(), np.array([1.0]), left, right, False)


@pytest.mark.parametrize(
    "state",
    [
        dict(status="running", cleanup_verified=False),
        dict(status="complete", cleanup_verified=True, completed={}, sessions={}),
    ],
)
def test_smoke_acceptance_rejects_incomplete_or_missing_owned_runs_before_data_reads(
    tmp_path, state
):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "gae_accept", Path(__file__).parents[1] / "scripts/accept_gae_smokes.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    state_path, source_path = tmp_path / "state.json", tmp_path / "source.json"
    state_path.write_text(json.dumps(state))
    source_path.write_text("{}")
    with pytest.raises(ValueError, match="owned"):
        script.accept(state_path, source_path, tmp_path / "nonexistent-data")

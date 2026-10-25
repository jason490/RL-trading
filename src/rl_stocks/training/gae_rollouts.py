"""Actual pre-update GAE buffer evidence; read-only checks never optimize."""

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from rl_stocks.core.files import write_once
from rl_stocks.training.curves import append_record
from rl_stocks.training.gae_spec import protocol, parameters, PROTOCOL_SHA256

ATOL = 2e-6
RTOL = 2e-5
FLOAT_FIELDS = {
    "rewards",
    "raw_rewards",
    "timeout_values",
    "values",
    "episode_starts",
    "returns",
    "advantages",
    "last_values",
}
BOOL_FIELDS = {"dones", "step_dones", "timeouts"}


def validate_arrays(arrays, arm, smoke=False):
    params = parameters(arm, smoke)
    n = params["n_steps"]
    gamma = params["gamma"]
    lam = params["gae_lambda"]
    if set(arrays) != FLOAT_FIELDS | BOOL_FIELDS:
        raise ValueError("exact actual GAE buffer fields required")
    for name, value in arrays.items():
        shape = (1,) if name in ("dones", "last_values") else (n, 1)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != (np.dtype("float32") if name in FLOAT_FIELDS else np.dtype("bool"))
            or not np.isfinite(value).all()
        ):
            raise ValueError("actual finite GAE field dtype/shape required")
    starts = arrays["episode_starts"][:, 0]
    dones = arrays["step_dones"][:, 0]
    timeouts = arrays["timeouts"][:, 0]
    if (
        not np.isin(starts, [0, 1]).all()
        or not np.array_equal(starts[1:], dones[:-1])
        or arrays["dones"][0] != dones[-1]
        or (timeouts & ~dones).any()
        or np.any(arrays["timeout_values"][:, 0][~timeouts] != 0)
    ):
        raise ValueError("actual episode/timeout boundary masks mismatch")
    adjusted = arrays["raw_rewards"].astype(float) + gamma * arrays["timeout_values"].astype(float)
    np.testing.assert_allclose(arrays["rewards"], adjusted, atol=ATOL, rtol=RTOL)
    rewards = arrays["rewards"][:, 0].astype(float)
    values = arrays["values"][:, 0].astype(float)
    last = float(arrays["last_values"][0])
    advantage = np.empty(n)
    following = 0.0
    for i in reversed(range(n)):
        continuation = not dones[-1] if i == n - 1 else not bool(starts[i + 1])
        next_value = last if i == n - 1 else values[i + 1]
        delta = rewards[i] + gamma * continuation * next_value - values[i]
        following = delta + gamma * lam * continuation * following
        advantage[i] = following
    expected = values + advantage
    np.testing.assert_allclose(arrays["advantages"][:, 0], advantage, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(arrays["returns"][:, 0], expected, atol=ATOL, rtol=RTOL)
    # Separate segmented dot products do not use the GAE recurrence or values
    # inside a segment. Timeout rewards already include their terminal bootstrap.
    mc = np.empty(n)
    boundaries = [0, *list(np.flatnonzero(starts[1:]) + 1), n]
    for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        tail = last if end == n and not dones[-1] else 0.0
        for i in range(begin, end):
            mc[i] = np.dot(gamma ** np.arange(end - i), rewards[i:end]) + gamma ** (end - i) * tail
    if arm == "mc_trace":
        np.testing.assert_allclose(arrays["returns"][:, 0], mc, atol=ATOL, rtol=RTOL)
    return dict(
        n_steps=n,
        gamma=gamma,
        gae_lambda=lam,
        episode_starts=int(starts.sum()),
        timeouts=int(timeouts.sum()),
        last_done=bool(dones[-1]),
        all_fields_finite=True,
        maximum_timeout_reward_error=float(np.max(np.abs(arrays["rewards"] - adjusted))),
        maximum_advantage_error=float(np.max(np.abs(arrays["advantages"][:, 0] - advantage))),
        maximum_return_error=float(np.max(np.abs(arrays["returns"][:, 0] - expected))),
        maximum_segmented_mc_return_error=float(np.max(np.abs(arrays["returns"][:, 0] - mc))),
        candidate_mc_identity_checked=arm == "mc_trace",
        atol=ATOL,
        rtol=RTOL,
    )


def read_arrays(raw):
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("duplicate rollout array fields")
        return {name: archive[name].copy() for name in archive.files}


class GaeRolloutAudit(BaseCallback):
    """Attach beside validation callback; never mutate collector locals or RNG."""

    def __init__(self, path, arm, smoke=False):
        super().__init__()
        parameters(arm, smoke)
        self.path = Path(path)
        self.arm = arm
        self.smoke = smoke
        self.records = []
        self.raw = []
        self.timeout = []
        self.done = []
        self.tails = []

    def _on_rollout_start(self):
        self.raw = []
        self.timeout = []
        self.done = []
        self.tails = []

    def _on_step(self):
        raw = np.asarray(self.locals["rewards"], dtype=np.float32).copy()
        dones = np.asarray(self.locals["dones"], dtype=bool).copy()
        infos = self.locals["infos"]
        if raw.shape != (1,) or dones.shape != (1,) or len(infos) != 1:
            raise ValueError("fixed one-env GAE collector required")
        mask = np.array([dones[0] and bool(infos[0].get("TimeLimit.truncated", False))], dtype=bool)
        tail = np.zeros(1, dtype=np.float32)
        if mask[0]:
            if "terminal_observation" not in infos[0]:
                raise ValueError("actual timeout observation required")
            state = self.locals["lstm_states"].vf
            before = tuple(s.clone() for s in state)
            with torch.no_grad():
                value = self.model.policy.predict_values(
                    self.model.policy.obs_to_tensor(infos[0]["terminal_observation"])[0],
                    state,
                    torch.tensor([False], dtype=torch.float32),
                )
            for old, current in zip(before, state, strict=True):
                torch.testing.assert_close(old, current, atol=0, rtol=0)
            tail[0] = float(value.item())
        self.raw.append(raw)
        self.done.append(dones)
        self.timeout.append(mask)
        self.tails.append(tail)
        return True

    def _on_rollout_end(self):
        model = self.model
        buffer = model.rollout_buffer
        params = parameters(self.arm, self.smoke)
        if (
            type(buffer).__name__ != "RecurrentRolloutBuffer"
            or buffer.n_envs != 1
            or not buffer.full
            or buffer.generator_ready
            or buffer.pos != params["n_steps"]
            or len(self.raw) != params["n_steps"]
            or model.gamma != params["gamma"]
            or buffer.gamma != params["gamma"]
            or model.gae_lambda != params["gae_lambda"]
            or buffer.gae_lambda != params["gae_lambda"]
            or type(model._n_updates) is not int
            or model._n_updates != len(self.records) * params["n_epochs"]
        ):
            raise ValueError("actual completed pre-update recurrent GAE buffer required")
        dones = np.asarray(model._last_episode_starts, dtype=bool).copy()
        state = model._last_lstm_states.vf
        before = tuple(s.clone() for s in state)
        rng = torch.get_rng_state().clone()
        with torch.no_grad():
            last = (
                model.policy.predict_values(
                    model.policy.obs_to_tensor(model._last_obs)[0],
                    state,
                    torch.as_tensor(dones, dtype=torch.float32),
                )
                .cpu()
                .numpy()
                .reshape(1)
            )
        for old, current in zip(before, state, strict=True):
            torch.testing.assert_close(old, current, atol=0, rtol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        arrays = {
            name: np.array(getattr(buffer, name), copy=True)
            for name in ["rewards", "values", "episode_starts", "returns", "advantages"]
        }
        arrays.update(
            raw_rewards=np.array(self.raw, dtype=np.float32),
            timeout_values=np.array(self.tails, dtype=np.float32),
            step_dones=np.array(self.done, dtype=bool),
            timeouts=np.array(self.timeout, dtype=bool),
            dones=dones,
            last_values=last,
        )
        record = validate_arrays(arrays, self.arm, self.smoke)
        steps = model.num_timesteps
        if type(steps) is not int or steps != (len(self.records) + 1) * params["n_steps"]:
            raise ValueError("ordered actual rollout steps required")
        memory = BytesIO()
        np.savez_compressed(memory, **arrays)
        raw = memory.getvalue()
        name = f"rollout-{steps:06d}.npz"
        write_once(self.path / "gae-rollouts" / name, raw)
        record.update(
            protocol_sha256=PROTOCOL_SHA256,
            arm=self.arm,
            steps=steps,
            rollout_index=len(self.records) + 1,
            arrays_file="gae-rollouts/" + name,
            arrays_sha256=hashlib.sha256(raw).hexdigest(),
            before_optimizer_update=True,
            sb3_nominal_epoch_counter=int(model._n_updates),
        )
        append_record(self.path / "gae-rollouts.jsonl", record)
        self.records.append(record)

    def summary(self):
        spec = protocol()
        target = spec["smoke_steps" if self.smoke else "steps_per_cell"]
        if not self.records or self.records[-1]["steps"] != target:
            raise ValueError("complete actual GAE rollout evidence required")
        return dict(
            protocol_sha256=PROTOCOL_SHA256,
            arm=self.arm,
            rollouts=len(self.records),
            transitions=target,
            all_checked_before_optimizer=True,
            all_actual_arrays_retained=True,
            maximum_return_error=max(r["maximum_return_error"] for r in self.records),
            maximum_advantage_error=max(r["maximum_advantage_error"] for r in self.records),
            candidate_mc_identity_checked=self.arm == "mc_trace",
        )

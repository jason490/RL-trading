"""Read-only actor-state restoration and native-direction Gaussian KL audit."""

import math

import torch
from torch.nn.functional import linear

from rl_stocks.evaluation.offpolicy_fit import fingerprint

SHAPES = {
    "log_std": (4,),
    "mlp_extractor.policy_net.0.weight": (64, 21),
    "mlp_extractor.policy_net.0.bias": (64,),
    "mlp_extractor.policy_net.2.weight": (64, 64),
    "mlp_extractor.policy_net.2.bias": (64,),
    "action_net.weight": (4, 64),
    "action_net.bias": (4,),
}


def distribution(state, observations):
    if set(state) != set(SHAPES) or any(
        not isinstance(state[k], torch.Tensor)
        or state[k].device.type != "cpu"
        or state[k].dtype != torch.float32
        or tuple(state[k].shape) != shape
        or not torch.isfinite(state[k]).all()
        for k, shape in SHAPES.items()
    ):
        raise ValueError("finite exact CPU float32 actor geometry required")
    if (
        not isinstance(observations, torch.Tensor)
        or observations.device.type != "cpu"
        or observations.dtype != torch.float32
        or observations.ndim != 2
        or observations.shape[1] != 21
        or not 1 <= len(observations) <= 1000
        or not torch.isfinite(observations).all()
    ):
        raise ValueError("bounded finite original rollout observations required")
    x = observations
    for prefix in ("mlp_extractor.policy_net.0", "mlp_extractor.policy_net.2"):
        x = torch.tanh(linear(x, state[prefix + ".weight"], state[prefix + ".bias"]))
    mean = linear(x, state["action_net.weight"], state["action_net.bias"])
    scale = state["log_std"].exp().expand_as(mean)
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("finite positive restored Gaussian required")
    return torch.distributions.Normal(mean, scale)


@torch.no_grad()
def verify_updates(states, records, *, algorithm, rollout):
    if (
        algorithm not in ("ppo", "trpo")
        or type(rollout) is not int
        or rollout not in (128, 1000)
        or not isinstance(states, list)
        or not 1 <= len(states) <= 8
        or len(states) != len(records)
    ):
        raise ValueError("bounded matching update evidence required")
    original = fingerprint(states)
    rows = []
    previous = None
    for index, (state, row) in enumerate(zip(states, records, strict=True)):
        if set(state) != {"before", "after", "observations"}:
            raise ValueError("exact saved update fields required")
        before, after, obs = state["before"], state["after"], state["observations"]
        old = distribution(before, obs)
        new = distribution(after, obs)
        left, right = fingerprint(before), fingerprint(after)
        if (
            len(obs) != rollout
            or row["transition"] != (index + 1) * rollout
            or row["actor_before_sha256"] != left
            or row["actor_after_sha256"] != right
            or type(row["actor_changed"]) is not bool
            or row["actor_changed"] != (left != right)
            or previous is not None and previous != left
        ):
            raise ValueError("saved actor identity/cadence/continuity mismatch")
        # SB3-Contrib2.7 uses KL(new || old), mean across samples AND actions.
        kl = float(torch.distributions.kl_divergence(new, old).mean())
        if not math.isfinite(kl) or kl < 0:
            raise ValueError("finite nonnegative independent KL required")
        if algorithm == "trpo":
            accepted = row["line_search_accepted"]
            if type(accepted) is not bool or (
                accepted and (left == right or kl >= 0.010001)
                or not accepted and (left != right or kl != 0)
                or not math.isfinite(row["kl"])
                or abs(kl - row["kl"]) > 1e-6
                or not 0 <= row["kl"] < 0.01
            ):
                raise ValueError("independent KL/rollback acceptance mismatch")
        rows.append({"transition": row["transition"], "independent_kl": kl})
        previous = right
    if fingerprint(states) != original:
        raise ValueError("audit mutated saved actor evidence")
    return {"updates": rows, "saved_state_unchanged": True, "training_performed": False}

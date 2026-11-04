import copy

import pytest
import torch

from rl_stocks.evaluation.offpolicy_fit import fingerprint
from rl_stocks.evaluation.trpo_updates import SHAPES, distribution, verify_updates


def evidence():
    # Synthetic corruption fixtures, never optimization or market replay.
    before = {k: torch.zeros(shape) for k, shape in SHAPES.items()}
    after = copy.deepcopy(before)
    after["action_net.bias"].fill_(0.01)
    obs = torch.zeros((128, 21))
    state = {"before": before, "after": after, "observations": obs}
    row = {
        "transition": 128, "actor_changed": True,
        "actor_before_sha256": fingerprint(before),
        "actor_after_sha256": fingerprint(after),
        "line_search_accepted": True, "kl": 0.00005,
    }
    return [state], [row]


def test_independent_restoration_and_immutable_state():
    states, rows = evidence()
    proof = verify_updates(states, rows, algorithm="trpo", rollout=128)
    assert proof["saved_state_unchanged"] and not proof["training_performed"]
    assert proof["updates"][0]["independent_kl"] == pytest.approx(0.00005, abs=1e-7)


@pytest.mark.parametrize("corruption", ["hash", "rollback", "kl", "continuity", "observations"])
def test_reject_corrupted_saved_evidence(corruption):
    states, rows = evidence()
    if corruption == "hash":
        rows[0]["actor_before_sha256"] = "wrong"
    elif corruption == "rollback":
        rows[0]["line_search_accepted"] = False
    elif corruption == "kl":
        rows[0]["kl"] = 0.005
    elif corruption == "continuity":
        states.append(copy.deepcopy(states[0]))
        rows.append(dict(rows[0], transition=256))
    else:
        states[0]["observations"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        verify_updates(states, rows, algorithm="trpo", rollout=128)


def test_gaussian_direction_and_action_mean_reduction():
    states, _ = evidence()
    state = states[0]
    state["after"]["log_std"].fill_(0.2)
    new = distribution(state["after"], state["observations"])
    old = distribution(state["before"], state["observations"])
    forward = torch.distributions.kl_divergence(new, old).mean()
    reverse = torch.distributions.kl_divergence(old, new).mean()
    assert abs(float(forward - reverse)) > 0.001
    assert new.mean.shape == (128, 4)


def test_restored_distribution_matches_native_without_learning():
    from test_trpo_feasibility import ContractEnv
    from rl_stocks.training.trpo_feasibility import create
    from rl_stocks.training.trpo_audit import clone_actor

    model = create("trpo", ContractEnv(), smoke=True)
    obs = torch.linspace(-2, 2, 128 * 21).reshape(128, 21)
    original = fingerprint(model.policy.state_dict())
    with torch.no_grad():
        native = model.policy.get_distribution(obs).distribution
        restored = distribution(clone_actor(model.policy), obs)
    torch.testing.assert_close(native.loc, restored.loc, rtol=0, atol=0)
    torch.testing.assert_close(native.scale, restored.scale, rtol=0, atol=0)
    assert fingerprint(model.policy.state_dict()) == original
    assert model.num_timesteps == 0 and not model.policy.optimizer.state

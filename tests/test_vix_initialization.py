from copy import deepcopy

from gymnasium import Env, spaces
import numpy as np
import pytest
import torch

from rl_stocks.rl.registry.factory import create
from rl_stocks.training.shared_initialization import tensor_fingerprint
from rl_stocks.training.vix_initialization import (
    source_initialization,
    map_initial_tensors,
    validate_mapping,
    INPUT_KEYS,
    seeded_fingerprints,
)
from rl_stocks.training.vix_spec import parameters


class ModelSpaces(Env):
    observation_space = spaces.Box(-np.inf, np.inf, (23,), dtype=np.float32)
    action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)


@pytest.mark.parametrize("seed", [11, 23, 37])
def test_zero_embedding_preserves_all_original_tensors_and_parameter_ownership(seed):
    torch.set_num_threads(1)
    _, original, _ = source_initialization(seed)
    model = create("recurrent_ppo", ModelSpaces(), seed=seed, device="cpu", overrides=parameters())
    owners = [id(p) for p in model.policy.parameters()]
    assert seeded_fingerprints() == seeded_fingerprints(seed)
    rng = torch.get_rng_state().clone()
    proof = map_initial_tensors(model, original)
    actual = {k: v.detach().numpy() for k, v in model.policy.state_dict().items()}
    assert validate_mapping(original, actual) == proof
    assert proof["parameter_counts"] == dict(legacy=61513, augmented=62537)
    assert owners == [id(p) for p in model.policy.parameters()]
    assert seeded_fingerprints() == seeded_fingerprints(seed)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not model.policy.optimizer.state and model.num_timesteps == 0 and model._n_updates == 0
    with pytest.raises(ValueError):
        map_initial_tensors(model, original)


def test_bad_source_rejects_before_model_mutation():
    _, original, _ = source_initialization(11)
    model = create("recurrent_ppo", ModelSpaces(), seed=11, device="cpu", overrides=parameters())
    before = tensor_fingerprint(
        {k: v.detach().numpy() for k, v in model.policy.state_dict().items()}
    )
    bad = deepcopy(original)
    bad[INPUT_KEYS[0]] = bad[INPUT_KEYS[0]][:, :20]
    with pytest.raises(ValueError):
        map_initial_tensors(model, bad)
    assert (
        tensor_fingerprint({k: v.detach().numpy() for k, v in model.policy.state_dict().items()})
        == before
    )
    model.num_timesteps = 1
    with pytest.raises(ValueError):
        map_initial_tensors(model, original)


def test_zero_embedding_initial_predictions_ignore_new_context():
    _, original, _ = source_initialization(11)
    model = create("recurrent_ppo", ModelSpaces(), seed=11, device="cpu", overrides=parameters())
    map_initial_tensors(model, original)
    base = np.linspace(-0.5, 0.5, 23, dtype=np.float32)
    changed = base.copy()
    changed[7:9] = [100.0, -100.0]
    left, state = model.predict(
        base, state=None, episode_start=np.array([True]), deterministic=True
    )
    right, other = model.predict(
        changed, state=None, episode_start=np.array([True]), deterministic=True
    )
    np.testing.assert_array_equal(left, right)
    for a, b in zip(state, other, strict=True):
        np.testing.assert_array_equal(a, b)
    assert model.num_timesteps == 0 and not model.policy.optimizer.state

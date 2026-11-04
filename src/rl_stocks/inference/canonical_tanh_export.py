"""Export bounded trusted native PPO/TRPO checkpoints without optimization."""

import hashlib
import json
from pathlib import Path
import random
import zipfile

import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from sb3_contrib import TRPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor, MlpExtractor

from rl_stocks.core.contracts import digest
from rl_stocks.inference.canonical_actor import encode32
from rl_stocks.inference.canonical_math import CONTRACT
from rl_stocks.inference.canonical_tanh_policy import CanonicalTanhPolicy, SCHEMA, OUTPUT


def _layer(layer, inputs, outputs):
    if (
        type(layer) is not nn.Linear
        or layer.in_features != inputs
        or layer.out_features != outputs
        or layer.bias is None
        or tuple(layer.weight.shape) != (outputs, inputs)
        or tuple(layer.bias.shape) != (outputs,)
    ):
        raise ValueError("fixed native actor layer required")
    return {
        "weights": [[encode32(v) for v in row] for row in layer.weight.detach().tolist()],
        "biases": [encode32(v) for v in layer.bias.detach().tolist()],
    }


def export_policy(source, output, *, algorithm):
    if algorithm not in ("ppo", "trpo"):
        raise ValueError("native PPO/TRPO source required")
    source, output = Path(source), Path(output)
    if not source.is_file() or source.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("bounded trusted project checkpoint required")
    with zipfile.ZipFile(source) as archive:
        if archive.getinfo("data").file_size > 2 * 1024 * 1024:
            raise ValueError("bounded native checkpoint metadata required")
        metadata = json.loads(archive.read("data"))
        if (
            algorithm == "trpo"
            and ("cg_max_steps" not in metadata or "n_epochs" in metadata)
            or algorithm == "ppo"
            and ("n_epochs" not in metadata or "cg_max_steps" in metadata)
        ):
            raise ValueError("checkpoint algorithm metadata mismatch")
    raw = source.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            model = (PPO if algorithm == "ppo" else TRPO).load(source, device="cpu")
            policy = model.policy
            if (
                type(model) is not (PPO if algorithm == "ppo" else TRPO)
                or type(policy) is not ActorCriticPolicy
                or policy.use_sde
                or policy.squash_output
                or type(policy.features_extractor) is not FlattenExtractor
                or type(policy.mlp_extractor) is not MlpExtractor
                or type(policy.action_dist) is not DiagGaussianDistribution
                or model.observation_space.shape != (21,)
                or model.action_space.shape != (4,)
                or model.observation_space.dtype != np.float32
                or model.action_space.dtype != np.float32
                or not np.array_equal(model.action_space.low, np.full(4, -1.0))
                or not np.array_equal(model.action_space.high, np.ones(4))
            ):
                raise ValueError("native full21-field clipped-Gaussian contract required")
            if any(
                t.dtype != torch.float32 or not torch.isfinite(t).all()
                for t in policy.state_dict().values()
            ):
                raise ValueError("finite binary32 policy state required")
            layers = policy.mlp_extractor.policy_net
            if (
                type(layers) is not nn.Sequential
                or len(layers) != 4
                or type(layers[1]) is not nn.Tanh
                or type(layers[3]) is not nn.Tanh
            ):
                raise ValueError("exact64x64 tanh actor required")
            body = {
                "schema": SCHEMA,
                "arithmetic": CONTRACT,
                "algorithm": algorithm,
                "input_size": 21,
                "hidden_sizes": [64, 64],
                "output_size": 4,
                "output_transform": OUTPUT,
                "source_model_sha256": source_sha,
                "layers": [
                    _layer(layers[0], 21, 64),
                    _layer(layers[2], 64, 64),
                    _layer(policy.action_net, 64, 4),
                ],
            }
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    if source.read_bytes() != raw:
        raise ValueError("checkpoint changed during export")
    envelope = {"actor": body, "sha256": digest(body)}
    CanonicalTanhPolicy(envelope, expected_source=source_sha, expected_algorithm=algorithm)
    with output.open("x") as stream:
        json.dump(envelope, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    return output

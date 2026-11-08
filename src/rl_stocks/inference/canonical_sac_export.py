"""Read-only export of trusted project standard SAC21/64x64 native checkpoints."""

import json
from pathlib import Path
import random
import zipfile

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import Actor, SACPolicy
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.torch_layers import FlattenExtractor
import torch
from torch import nn

from rl_stocks.core.contracts import digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.inference.canonical_actor import OUTPUT
from rl_stocks.inference.canonical_export import _layer
from rl_stocks.inference.canonical_math import CONTRACT
from rl_stocks.inference.canonical_sac_policy import CanonicalSACPolicy, SCHEMA


def export_policy(source, output):
    source, output = Path(source), Path(output)
    if not source.is_file() or source.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("bounded trusted project SAC checkpoint required")
    with zipfile.ZipFile(source) as archive:
        if archive.getinfo("data").file_size > 2 * 1024 * 1024:
            raise ValueError("bounded native metadata required")
    original = checksum(source)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            model = SAC.load(source, device="cpu")
            actor = model.actor
            if (type(model.policy) is not SACPolicy or type(actor) is not Actor
                    or model.policy.net_arch != [64, 64] or actor.use_sde
                    or not actor.squash_output
                    or type(actor.features_extractor) is not FlattenExtractor
                    or type(actor.action_dist) is not SquashedDiagGaussianDistribution
                    or model.observation_space.shape != (21,) or model.action_space.shape != (4,)
                    or model.observation_space.dtype != np.float32
                    or model.action_space.dtype != np.float32
                    or not np.array_equal(model.action_space.low, np.full(4, -1.))
                    or not np.array_equal(model.action_space.high, np.ones(4))):
                raise ValueError("native standard SAC21/64x64 contract required")
            if any(t.dtype != torch.float32 or not torch.isfinite(t).all()
                   for t in actor.state_dict().values()):
                raise ValueError("finite binary32 standard SAC actor required")
            layers = actor.latent_pi
            if (type(layers) is not nn.Sequential or len(layers) != 4
                    or type(layers[1]) is not nn.ReLU or type(layers[3]) is not nn.ReLU):
                raise ValueError("standard SAC two-ReLU hidden layers required")
            # Validate stochastic head geometry too; deterministic reference uses mu.
            _layer(actor.log_std, 64, 4)
            body = {"schema": SCHEMA, "arithmetic": CONTRACT, "algorithm": "sac",
                    "input_size": 21, "hidden_sizes": [64, 64], "output_size": 4,
                    "source_model_sha256": original, "output_transform": OUTPUT,
                    "layers": [_layer(layers[0], 21, 64), _layer(layers[2], 64, 64),
                               _layer(actor.mu, 64, 4)]}
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    if checksum(source) != original:
        raise ValueError("native source checkpoint changed during export")
    envelope = {"actor": body, "sha256": digest(body)}
    CanonicalSACPolicy(envelope, expected_source=original)
    with output.open("x") as stream:
        json.dump(envelope, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    return output

"""Export trusted project SAC checkpoints; never load untrusted pickle archives."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import FlattenExtractor
import torch
from torch import nn

from rl_stocks.core.contracts import digest
from rl_stocks.inference.canonical_actor import (
    CanonicalActor, OUTPUT, SCHEMA, encode32,
)
from rl_stocks.inference.canonical_math import CONTRACT
from rl_stocks.rl.sac.task_policy import RoutedHead, TaskActor, TaskSACPolicy


def _layer(layer, inputs, outputs):
    if (type(layer) is not nn.Linear or layer.in_features != inputs
            or layer.out_features != outputs or layer.bias is None
            or tuple(layer.weight.shape) != (outputs, inputs)
            or tuple(layer.bias.shape) != (outputs,)):
        raise ValueError('unsupported actor layer geometry')
    return {'weights': [[encode32(v) for v in row] for row in layer.weight.detach().tolist()],
            'biases': [encode32(v) for v in layer.bias.detach().tolist()]}


def export_actor(source, output):
    """Write a fresh checksummed reference actor, preserving host global RNGs.

    SAC.load executes checkpoint pickle content: only task-owned trusted files
    are supported. No model learning, optimizer step or normalization fit occurs.
    """
    source, output = Path(source), Path(output)
    if not source.is_file() or source.stat().st_size > 32 * 1024 * 1024:
        raise ValueError('bounded trusted SAC checkpoint required')
    raw = source.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            model = SAC.load(source, device='cpu')
            actor = model.actor
            if (type(model.policy) is not TaskSACPolicy or type(actor) is not TaskActor
                    or actor.task_role != model.policy.task_role or actor.use_sde
                    or type(actor.features_extractor) is not FlattenExtractor
                    or model.observation_space.shape != (22,)
                    or model.action_space.shape != (4,)
                    or not np.array_equal(model.action_space.low, np.full(4, -1.))
                    or not np.array_equal(model.action_space.high, np.ones(4))):
                raise ValueError('unsupported task actor contract')
            if any(t.dtype != torch.float32 or not torch.isfinite(t).all()
                   for t in actor.state_dict().values()):
                raise ValueError('finite binary32 actor weights required')
            layers = actor.latent_pi
            if (type(layers) is not nn.Sequential or len(layers) != 4
                    or type(layers[1]) is not nn.ReLU or type(layers[3]) is not nn.ReLU):
                raise ValueError('unsupported actor hidden layers')
            hidden = [_layer(layers[0], 22, 64), _layer(layers[2], 64, 64)]
            if actor.task_role == 'tagged_routed':
                if type(actor.mu) is not RoutedHead:
                    raise ValueError('routed mean heads required')
                heads = {name: _layer(getattr(actor.mu, name), 64, 4) for name in ('stock', 'etf')}
            else:
                heads = {'shared': _layer(actor.mu, 64, 4)}
            body = {'schema': SCHEMA, 'arithmetic': CONTRACT, 'role': actor.task_role,
                    'input_size': 22, 'hidden_sizes': [64, 64], 'output_size': 4,
                    'source_model_sha256': source_sha, 'output_transform': OUTPUT,
                    'layers': hidden, 'heads': heads}
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    if source.read_bytes() != raw:
        raise ValueError('source checkpoint changed during export')
    envelope = {'actor': body, 'sha256': digest(body)}
    CanonicalActor(envelope, expected_source=source_sha)
    with output.open('x') as stream:
        json.dump(envelope, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    return output

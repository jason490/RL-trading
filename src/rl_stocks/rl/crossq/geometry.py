"""Actual CrossQ geometry and immutable normalization during frozen prediction."""

import numpy as np
from sb3_contrib import CrossQ
from sb3_contrib.common.torch_layers import BatchRenorm
import torch

from rl_stocks.core.contracts import canonical
from rl_stocks.training.shared_initialization import tensor_fingerprint


def layers(model):
    return {name: layer for name, layer in model.policy.named_modules() if isinstance(layer, BatchRenorm)}


def model_spec(model):
    normalization = layers(model)
    if (type(model) is not CrossQ or model.observation_space.shape != (21,)
        or model.action_space.shape != (4,) or model.policy.net_arch != [64, 64]
        or hasattr(model, 'critic_target') or model.policy_delay != 3
        or model.use_sde is not False or len(model.critic.q_networks) != 2
        or len(normalization) != 9
        or sum(p.numel() for p in model.actor.parameters()) != 6386
        or sum(p.numel() for p in model.critic.parameters()) != 12390):
        raise ValueError('actual fixed CrossQ policy/critic geometry required')
    for layer in normalization.values():
        if layer.warmup_steps != 1000 or layer.eps != 0.001 or layer.momentum != 0.01:
            raise ValueError('fixed CrossQ normalization settings required')
    return dict(version=1, kind='crossq_batch_normalized_twin_critics', actor_parameters=6386,
                critic_parameters=12390, target_critic_present=False, linear_widths=[64, 64],
                normalization_layers=9, renorm_warmup_steps=1000,
                normalization_eps=0.001, normalization_momentum=0.01, policy_delay=3)


def normalization_record(model):
    result = {}
    for name, layer in layers(model).items():
        state = {k: v.detach().cpu().numpy() for k, v in layer.state_dict().items()}
        if (int(layer.steps) < 0 or layer.steps.dtype != torch.long
            or any(not np.isfinite(v).all() for v in state.values())
            or bool((layer.ra_var < 0).any())):
            raise ValueError('finite nonnegative CrossQ running state required')
        result[name] = dict(features=layer.num_features, steps=int(layer.steps),
                            state_sha256=tensor_fingerprint(state))
    return result


def validate_loaded(model, declared, recorded, algorithm):
    if algorithm != 'crossq':
        if isinstance(model, CrossQ) or declared is not None or recorded is not None:
            raise ValueError('algorithm and CrossQ normalization contract mismatch')
        return None
    actual = model_spec(model)
    if canonical(declared) != canonical(actual) or canonical(recorded) != canonical(normalization_record(model)):
        raise ValueError('explicit saved CrossQ geometry/normalization required')
    return actual


def frozen_predict(model, observation, expected):
    normalization = layers(model)
    if any(layer.training for layer in normalization.values()) or canonical(normalization_record(model)) != canonical(expected):
        raise ValueError('frozen CrossQ normalization changed before prediction')
    saved = {name: {k: v.clone() for k, v in layer.state_dict().items()} for name, layer in normalization.items()}
    try:
        action = model.predict(np.array(observation, dtype=np.float32, copy=True), deterministic=True)[0]
    finally:
        changed = any(layer.training for layer in normalization.values()) or any(not torch.equal(v, saved[name][k]) for name, layer in normalization.items() for k, v in layer.state_dict().items())
        if changed:
            for name, layer in normalization.items():
                layer.load_state_dict(saved[name], strict=True)
                layer.train(False)
            raise ValueError('prediction mutated frozen CrossQ normalization; restored original state')
    return action

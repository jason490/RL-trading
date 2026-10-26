"""Untrained common Linear mapping; normalization is native, never fitted."""

from sb3_contrib import CrossQ
from sb3_contrib.common.torch_layers import BatchRenorm
from stable_baselines3 import SAC
import torch

from rl_stocks.training.crossq_spec import parameters
from rl_stocks.training.shared_initialization import _untrained


def _linears(model):
    return {branch: [(name, layer) for name, layer in getattr(model, branch).named_modules()
                     if isinstance(layer, torch.nn.Linear)] for branch in ('actor', 'critic')}


def _native_normalization(model):
    expected = parameters('crossq')['policy_kwargs']
    layers = [(name, layer) for name, layer in model.policy.named_modules()
              if isinstance(layer, BatchRenorm)]
    if len(layers) != 9:
        raise ValueError('nine native CrossQ normalization layers required')
    for _, layer in layers:
        if (layer.eps != expected['batch_norm_eps']
            or layer.momentum != expected['batch_norm_momentum']
            or layer.warmup_steps != expected['renorm_warmup_steps']
            or int(layer.steps) != 0
            or not torch.equal(layer.ra_mean, torch.zeros_like(layer.ra_mean))
            or not torch.equal(layer.ra_var, torch.ones_like(layer.ra_var))
            or not torch.equal(layer.scale, torch.ones_like(layer.scale))
            or not torch.equal(layer.bias, torch.zeros_like(layer.bias))):
            raise ValueError('unfitted native CrossQ normalization state required')
    return layers


def _validate(sac, crossq):
    if type(sac) is not SAC or type(crossq) is not CrossQ:
        raise ValueError('native SAC and CrossQ scaffolds required')
    for arm, model in [('sac', sac), ('crossq', crossq)]:
        _untrained(model)
        spec = parameters(arm)
        if (model.observation_space.shape != (21,) or model.action_space.shape != (4,)
            or model.policy.net_arch != [64, 64]
            or len(model.critic.q_networks) != 2
            or model.seed != sac.seed or type(model.seed) is not int
            or model.seed not in (11, 23, 37)
            or model.log_ent_coef is None or model.log_ent_coef.numel() != 1
            or float(model.log_ent_coef.detach()) != 0):
            raise ValueError('fixed21-input/4-action/full64 seeded geometry and initial entropy required')
        counts = (6088, 11778) if arm == 'sac' else (6386, 12390)
        if tuple(sum(p.numel() for p in getattr(model, branch).parameters()) for branch in ('actor', 'critic')) != counts:
            raise ValueError('fixed actual parameter counts required')
        for branch in ('actor', 'critic'):
            optimizer = getattr(model, branch).optimizer
            actual = [id(p) for group in optimizer.param_groups for p in group['params']]
            expected_ids = [id(p) for p in getattr(model, branch).parameters()]
            if len(actual) != len(set(actual)) or set(actual) != set(expected_ids):
                raise ValueError('optimizer must own every live parameter exactly once')
            for group in optimizer.param_groups:
                expected = spec['policy_kwargs']['optimizer_kwargs']
                if (tuple(group['betas']) != expected['betas'] or group['eps'] != expected['eps']
                    or group['lr'] != spec['learning_rate'] or group['weight_decay'] != 0
                    or group['amsgrad'] is not False):
                    raise ValueError('native fixed actor/critic optimizer settings required')
    if not hasattr(sac, 'critic_target') or hasattr(crossq, 'critic_target'):
        raise ValueError('SAC target and CrossQ absent-target contract required')
    if any(not torch.equal(v, sac.critic_target.state_dict()[k]) for k, v in sac.critic.state_dict().items()):
        raise ValueError('untrained SAC target must equal live critic')
    if crossq.policy_delay != 3:
        raise ValueError('fixed CrossQ actor delay required')
    _native_normalization(crossq)
    left, right = _linears(sac), _linears(crossq)
    for branch, count in [('actor', 4), ('critic', 6)]:
        if len(left[branch]) != count or len(right[branch]) != count:
            raise ValueError('complete actor/head and twin critic Linear geometry required')
        for (_, source), (_, target) in zip(left[branch], right[branch], strict=True):
            if (source.bias is None or target.bias is None
                or source.weight.shape != target.weight.shape
                or source.bias.shape != target.bias.shape
                or source.weight.dtype != target.weight.dtype
                or not torch.isfinite(source.weight).all() or not torch.isfinite(source.bias).all()):
                raise ValueError('finite same-shape/dtype Linear tensors required before mapping')
    return left, right


def map_linears(sac, crossq):
    """Validate all ownership/settings/geometry before copying any target tensor."""
    left, right = _validate(sac, crossq)
    ownership = [id(p) for p in crossq.policy.parameters()]
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        for branch in left:
            for (_, source), (_, target) in zip(left[branch], right[branch], strict=True):
                target.weight.copy_(source.weight)
                target.bias.copy_(source.bias)
    result = validate_mapping(sac, crossq)
    if ownership != [id(p) for p in crossq.policy.parameters()] or not torch.equal(rng, torch.get_rng_state()):
        raise ValueError('mapping changed parameter ownership or sampled randomness')
    result['parameter_ownership_preserved'] = True
    return result


def validate_mapping(sac, crossq):
    left, right = _validate(sac, crossq)
    mapping = []
    for branch in left:
        for (source_name, source), (target_name, target) in zip(left[branch], right[branch], strict=True):
            if not torch.equal(source.weight, target.weight) or not torch.equal(source.bias, target.bias):
                raise ValueError('all common Linear weights and biases must match exactly')
            mapping.append(dict(branch=branch, source=source_name, target=target_name,
                                weight_shape=list(source.weight.shape)))
    return dict(common_linears_exact=True, native_normalization_unchanged=True,
                optimizer_ownership_valid=True, mapping=mapping,
                actor_parameters=6386, critic_parameters=12390,
                optimizer_steps=0, transitions=0)

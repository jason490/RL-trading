"""Causal observation-only GRU histories for a prospective recurrent SAC adapter.

These are forward helpers, not a registered learner. Each encoder reconstructs
its own state from the sampled prefix, never a collection-time cached state.
Finite burn-in from a nonzero episode step remains an approximation.
"""
from dataclasses import dataclass
from collections.abc import Mapping

import torch
from torch import nn

from rl_stocks.training.recurrent_replay import _integer

ROLES = ('actor', 'critic_1', 'critic_2', 'target_critic_1', 'target_critic_2')


@dataclass(frozen=True)
class AlignedHistory:
    current: torch.Tensor
    next: torch.Tensor
    burn_in_hidden: torch.Tensor
    final_hidden: torch.Tensor


def _validate(encoder, observations, burn_in, track_grad):
    if not isinstance(encoder, nn.GRU) or not encoder.batch_first or encoder.bidirectional:
        raise ValueError('unidirectional batch-first torch GRU required')
    if encoder.dropout != 0:
        raise ValueError('dropout is disabled for explicit history parity')
    if type(track_grad) is not bool:
        raise ValueError('explicit gradient-tracking boolean required')
    _integer('burn_in', burn_in)
    if (not isinstance(observations, torch.Tensor) or observations.ndim != 3
            or observations.dtype != torch.float32 or observations.shape[0] < 1
            or observations.shape[1] < burn_in + 2
            or observations.shape[2] != encoder.input_size
            or not bool(torch.isfinite(observations).all())):
        raise ValueError('finite float32 [B,burn_in+unroll+1,D] observations required')
    if any(p.device != observations.device or p.dtype != observations.dtype
           for p in encoder.parameters()):
        raise ValueError('encoder/input device and dtype must agree')


def encode_history(encoder: nn.GRU, observations: torch.Tensor, *, burn_in: int,
                   track_grad: bool = False) -> AlignedHistory:
    """Zero-initialize each call; align t/t+1 by shifting ONE causal forward pass.

    Prefix forward is always no-grad. Its state is detached before the suffix.
    Explicit tracking supports future Colab learning; no backward/update occurs
    here. A no-grad outer context still disables tracking. Targets use False.
    """
    _validate(encoder, observations, burn_in, track_grad)
    hidden = observations.new_zeros(encoder.num_layers, observations.shape[0], encoder.hidden_size)
    if burn_in:
        with torch.no_grad():
            _, hidden = encoder(observations[:, :burn_in], hidden)
    hidden = hidden.detach()
    # Detach inputs as well: replay never propagates into collection/preprocessing.
    suffix = observations[:, burn_in:].detach()
    with torch.set_grad_enabled(torch.is_grad_enabled() and track_grad):
        outputs, final = encoder(suffix, hidden)
    return AlignedHistory(outputs[:, :-1], outputs[:, 1:], hidden, final)


def sac_histories(encoders: Mapping[str, nn.GRU], observations: torch.Tensor, *,
                  burn_in: int, track_grad: bool = False) -> dict[str, AlignedHistory]:
    """Independent actor/twin-critic/target histories; targets never track gradients.

    Action values enter downstream Q heads, not these observation-only memories.
    No target actor is assumed by SAC. Shared/tied encoder storage is rejected.
    """
    if not isinstance(encoders, Mapping) or set(encoders) != set(ROLES):
        raise ValueError('exactly five named SAC recurrent encoders required')
    if type(track_grad) is not bool:
        raise ValueError('explicit gradient-tracking boolean required')
    seen = set()
    for role in ROLES:
        encoder = encoders[role]
        _validate(encoder, observations, burn_in, track_grad)
        storage = {(p.device, p.untyped_storage().data_ptr()) for p in encoder.parameters()}
        if seen & storage:
            raise ValueError('actor, critics and targets require independent parameter storage')
        seen.update(storage)
    return {role: encode_history(encoders[role], observations, burn_in=burn_in,
                                 track_grad=track_grad and not role.startswith('target_'))
            for role in ROLES}

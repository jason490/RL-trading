"""Recurrent SAC loss path, audited against installed SB3 2.7 equations.

No backward, optimizer or target update is performed here. Critic and actor
objectives are separate so the future learner can recompute actor Q values AFTER
its critic update, matching SB3's ordering. No padded sequences are supported.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math

import numpy as np
import torch

from rl_stocks.training.recurrent_history import encode_history
from rl_stocks.training.recurrent_payload import SequenceBatch
from rl_stocks.training.recurrent_replay import _integer


@dataclass(frozen=True)
class ReplayTensors:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    bootstrap: torch.Tensor
    burn_in: int


def replay_tensors(batch: SequenceBatch, device='cpu') -> ReplayTensors:
    if not isinstance(batch, SequenceBatch):
        raise ValueError('SequenceBatch required')
    _integer('burn_in', batch.burn_in)
    obs, actions = batch.observations, batch.actions
    if (obs.ndim != 3 or actions.ndim != 3 or obs.shape[0] < 1 or actions.shape[-1] != 4
            or obs.shape[:2] != (actions.shape[0], actions.shape[1] + 1)
            or obs.shape[2] < 1 or actions.shape[1] <= batch.burn_in):
        raise ValueError('replay tensor geometry mismatch')
    shape = actions.shape[:2]
    for value in (obs, actions, batch.rewards):
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('finite float32 replay fields required')
    if batch.rewards.shape != shape or np.any(np.abs(actions) > 1):
        raise ValueError('reward/action geometry or range mismatch')
    for mask in (batch.loss_mask, batch.bootstrap_mask, batch.reset_after):
        if mask.dtype != bool or mask.shape != shape:
            raise ValueError('boolean mask geometry mismatch')
    expected = np.arange(shape[1]) >= batch.burn_in
    if (not np.all(batch.loss_mask == expected) or np.any(batch.reset_after[:, :-1])
            or np.any(~batch.bootstrap_mask & ~batch.reset_after)):
        raise ValueError('unpadded within-episode loss/bootstrap/reset contract required')
    return ReplayTensors(torch.tensor(obs, device=device), torch.tensor(actions, device=device),
                         torch.tensor(batch.rewards, device=device),
                         torch.tensor(batch.bootstrap_mask, device=device), batch.burn_in)


def _scalar(value, name):
    if (not isinstance(value, torch.Tensor) or value.numel() != 1 or value.ndim > 1
            or not bool(torch.isfinite(value))):
        raise ValueError(f'finite scalar {name} required')


def bellman_target(rewards, bootstrap, next_q1, next_q2, next_log_prob, alpha, gamma):
    if type(gamma) not in (float, int) or not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError('discount must be finite in [0,1]')
    _scalar(alpha, 'alpha')
    if bool(alpha < 0) or bootstrap.dtype != torch.bool:
        raise ValueError('nonnegative alpha and boolean bootstrap required')
    if any(v.shape != rewards.shape for v in (bootstrap, next_q1, next_q2, next_log_prob)):
        raise ValueError('Bellman field shapes must agree')
    with torch.no_grad():
        result = rewards + gamma * bootstrap * (torch.minimum(next_q1, next_q2) - alpha * next_log_prob)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('nonfinite Bellman target')
    return result


def temperature_loss(log_alpha, log_prob, target_entropy=-4.):
    _scalar(log_alpha, 'log alpha')
    if (not math.isfinite(target_entropy) or log_prob.numel() == 0
            or not bool(torch.isfinite(log_prob).all())):
        raise ValueError('finite entropy target/log probabilities required')
    return -(log_alpha * (log_prob + target_entropy).detach()).mean()


@contextmanager
def frozen_parameters(module):
    parameters = list(module.parameters())
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def critic_objective(networks, replay: ReplayTensors, alpha, *, gamma=.99, next_noise=None):
    obs, burn = replay.observations, replay.burn_in
    with torch.no_grad():
        actor = encode_history(networks.actor.memory, obs, burn_in=burn)
        actions, log_prob = networks.actor.sample(actor.next, next_noise)
        next_q = []
        for target in networks.targets:
            history = encode_history(target.memory, obs, burn_in=burn)
            next_q.append(target(history.next, actions))
        target_values = bellman_target(replay.rewards[:, burn:], replay.bootstrap[:, burn:],
                                      *next_q, log_prob, alpha.detach(), gamma)
    current_q = []
    for critic in networks.critics:
        history = encode_history(critic.memory, obs, burn_in=burn, track_grad=True)
        current_q.append(critic(history.current, replay.actions[:, burn:]))
    loss = .5 * sum((q - target_values).square().mean() for q in current_q)
    _scalar(loss, 'critic loss')
    return loss, target_values, tuple(current_q)


def actor_objective(networks, replay: ReplayTensors, alpha, *, noise=None):
    _scalar(alpha, 'alpha')
    if bool(alpha < 0):
        raise ValueError('nonnegative alpha required')
    obs, burn = replay.observations, replay.burn_in
    actor = encode_history(networks.actor.memory, obs, burn_in=burn, track_grad=True)
    actions, log_prob = networks.actor.sample(actor.current, noise)
    # Freeze critic weights while retaining Q's derivative with respect to action.
    with frozen_parameters(networks.critics):
        q_values = []
        for critic in networks.critics:
            history = encode_history(critic.memory, obs, burn_in=burn)
            q_values.append(critic(history.current, actions))
        loss = (alpha.detach() * log_prob - torch.minimum(*q_values)).mean()
    _scalar(loss, 'actor loss')
    return loss, log_prob, actions

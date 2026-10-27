"""Bounded CPU Colab-only recurrent SAC updates; never a generic training route."""
import os
from pathlib import Path
import sys

import torch

from rl_stocks.training.recurrent_history import encode_history
from rl_stocks.training.recurrent_replay import _integer
from rl_stocks.training.recurrent_sac_losses import (
    actor_objective, critic_objective, replay_tensors, temperature_loss,
)


def require_colab_cpu():
    if (sys.version_info[:2] != (3, 13) or not os.environ.get('COLAB_RELEASE_TAG')
            or Path.cwd() != Path('/content/rl-stocks') or torch.cuda.is_available()):
        raise RuntimeError('recurrent SAC updates require default Python3.13 in owned Colab CPU')


class RecurrentSACLearner:
    def __init__(self, networks, *, max_updates: int, learning_rate=.0003, gamma=.99, tau=.005):
        _integer('max_updates', max_updates, 1)
        for name, value, low, high in [('learning rate', learning_rate, 0, 1),
                                       ('gamma', gamma, 0, 1), ('tau', tau, 0, 1)]:
            if (type(value) not in (float, int) or not torch.isfinite(torch.tensor(value))
                    or not low < value <= high):
                raise ValueError(f'invalid {name}')
        if any(p.device.type != 'cpu' for p in networks.parameters()):
            raise ValueError('CPU networks required')
        self.networks = networks
        self.max_updates = max_updates
        self.learning_rate, self.gamma, self.tau = learning_rate, gamma, tau
        self.log_alpha = torch.tensor(0., requires_grad=True)
        self.optimizers = dict(entropy=torch.optim.Adam([self.log_alpha], lr=learning_rate),
                               critic=torch.optim.Adam(networks.critics.parameters(), lr=learning_rate),
                               actor=torch.optim.Adam(networks.actor.parameters(), lr=learning_rate))
        self.updates = 0
        self.attempts = 0
        self.completed_steps = dict.fromkeys(self.optimizers, 0)
        self.failed = False
        self.last_order = []

    def _step(self, role, loss):
        if not loss.requires_grad or not bool(torch.isfinite(loss)):
            raise ValueError('finite differentiable loss required')
        optimizer = self.optimizers[role]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for group in optimizer.param_groups for p in group['params']]
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
            raise ValueError(f'finite gradients required for every {role} parameter')
        optimizer.step()
        self.completed_steps[role] += 1
        self.last_order.append(role)
        if any(not bool(torch.isfinite(p).all()) for p in params):
            raise ValueError(f'nonfinite {role} parameters after update')
        optimizer.zero_grad(set_to_none=True)

    def update(self, batch):
        require_colab_cpu()  # before tensor conversion, RNG changes, counters or any backward
        if self.failed or self.attempts >= self.max_updates or self.updates != self.attempts:
            raise RuntimeError('update allowance exhausted or prior partial update failed; no retry')
        replay = replay_tensors(batch)
        if replay.observations.shape[-1] != self.networks.observation_size:
            raise ValueError('observation schema mismatch')
        self.attempts += 1
        self.last_order = []
        try:
            # Draw current actions once for both temperature and actor phases.
            shape = (*replay.actions[:, replay.burn_in:].shape,)
            noise = torch.randn(shape)
            with torch.no_grad():
                alpha = self.log_alpha.detach().exp().clone()  # SB3 uses pre-entropy-step alpha
                history = encode_history(self.networks.actor.memory, replay.observations,
                                         burn_in=replay.burn_in)
                _, log_prob = self.networks.actor.sample(history.current, noise)
            entropy = temperature_loss(self.log_alpha, log_prob)
            self._step('entropy', entropy)
            critic, _, _ = critic_objective(self.networks, replay, alpha, gamma=self.gamma)
            self._step('critic', critic)
            # Reconstruct online critic histories with UPDATED critic parameters.
            actor, _, _ = actor_objective(self.networks, replay, alpha, noise=noise)
            self._step('actor', actor)
            with torch.no_grad():
                for source, target in zip(self.networks.critics.parameters(),
                                          self.networks.targets.parameters(), strict=True):
                    target.mul_(1 - self.tau).add_(source, alpha=self.tau)
            self.last_order.append('targets')
            self.updates += 1
            return dict(update=self.updates, actor_loss=float(actor.detach()),
                        critic_loss=float(critic.detach()), entropy_loss=float(entropy.detach()),
                        alpha_before=float(alpha), alpha_after=float(self.log_alpha.detach().exp()),
                        completed_steps=dict(self.completed_steps), order=list(self.last_order))
        except BaseException:
            self.failed = True
            raise
        finally:
            for optimizer in self.optimizers.values():
                optimizer.zero_grad(set_to_none=True)

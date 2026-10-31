"""Explicit-type SAC heads; SB3 retains distributions, losses and optimizers."""
from copy import deepcopy

import torch
from torch import nn
from stable_baselines3.sac.policies import Actor, SACPolicy
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import FlattenExtractor

ROLES = ('masked_shared', 'tagged_shared', 'tagged_routed')


def check_role(role):
    if role not in ROLES:
        raise ValueError('unknown task policy role')
    return role


def task_features(features, role):
    if features.ndim != 2 or features.shape[1] != 22:
        raise ValueError('task policy requires batch of full21 plus type')
    if not torch.isfinite(features).all():
        raise ValueError('nonfinite task observations')
    tag = features[:, -1:]
    expected = (tag == 0) if role == 'masked_shared' else ((tag == -1) | (tag == 1))
    if not expected.all():
        raise ValueError('invalid instrument type tag')
    return features, tag


class RoutedHead(nn.Module):
    """Copy an initialized head without RNG draws; no persistent batch state."""
    def __init__(self, head):
        super().__init__()
        self.stock = head
        self.etf = deepcopy(head)

    def forward(self, latent, tag):
        return torch.where(tag == 1, self.etf(latent), self.stock(latent))


class TaskActor(Actor):
    def __init__(self, *args, task_role, **kwargs):
        self.task_role = check_role(task_role)
        super().__init__(*args, **kwargs)
        if self.use_sde:
            raise ValueError('task policy requires squashed diagonal Gaussian')
        if self.task_role == 'tagged_routed':
            self.mu = RoutedHead(self.mu)
            self.log_std = RoutedHead(self.log_std)

    def get_action_dist_params(self, obs):
        features, tag = task_features(self.extract_features(obs, self.features_extractor),
                                      self.task_role)
        latent = self.latent_pi(features)
        if self.task_role == 'tagged_routed':
            mean, log_std = self.mu(latent, tag), self.log_std(latent, tag)
        else:
            mean, log_std = self.mu(latent), self.log_std(latent)
        return mean, torch.clamp(log_std, -20, 2), {}

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), 'task_role': self.task_role}


class TaskCritic(ContinuousCritic):
    def __init__(self, *args, task_role, **kwargs):
        self.task_role = check_role(task_role)
        super().__init__(*args, **kwargs)
        if self.task_role == 'tagged_routed':
            for q in self.q_networks:
                q[-1] = RoutedHead(q[-1])

    def forward(self, obs, actions):
        with torch.set_grad_enabled(not self.share_features_extractor):
            features, tag = task_features(
                self.extract_features(obs, self.features_extractor), self.task_role)
        inputs = torch.cat([features, actions], dim=1)
        if self.task_role == 'tagged_routed':
            return tuple(q[-1](q[:-1](inputs), tag) for q in self.q_networks)
        return tuple(q(inputs) for q in self.q_networks)

    def q1_forward(self, obs, actions):
        return self(obs, actions)[0]


class TaskSACPolicy(SACPolicy):
    def __init__(self, *args, task_role, **kwargs):
        self.task_role = check_role(task_role)
        if (kwargs.get('use_sde', False)
                or kwargs.get('features_extractor_class', FlattenExtractor) != FlattenExtractor
                or kwargs.get('share_features_extractor', False)):
            raise ValueError('task policy requires separate flattened actor/critic inputs')
        super().__init__(*args, **kwargs)
        if self.observation_space.shape != (22,) or self.action_space.shape != (4,):
            raise ValueError('task policy schema mismatch')
        if self.net_arch != [64, 64] or self.critic.n_critics != 2:
            raise ValueError('task policy requires two64 layers and twin critics')

    def make_actor(self, features_extractor=None):
        options = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return TaskActor(**options, task_role=self.task_role).to(self.device)

    def make_critic(self, features_extractor=None):
        options = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return TaskCritic(**options, task_role=self.task_role).to(self.device)

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), 'task_role': self.task_role}

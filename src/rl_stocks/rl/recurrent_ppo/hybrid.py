"""Categorical direction and Gaussian protection for recurrent PPO research.

The four-component environment contract is unchanged. Only direction is
categorical; protective samples retain SB3's usual Gaussian/clipping behavior.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from stable_baselines3.common.distributions import DiagGaussianDistribution
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy


class DirectionProtectionDistribution(DiagGaussianDistribution):
    """Joint independent direction/protection likelihood, with optional cash.

    Both variants have identical six-output parameter shapes. The binary control
    omits the cash logit from its categorical distribution, rather than sampling
    cash and replacing it afterward. Thus rollout and PPO likelihoods agree.
    """
    def __init__(self, cash_enabled):
        if type(cash_enabled) is not bool:
            raise ValueError('explicit boolean cash availability required')
        super().__init__(4)
        self.cash_enabled = cash_enabled

    def proba_distribution_net(self, latent_dim, log_std_init=0.0):
        return nn.Linear(latent_dim, 6), nn.Parameter(torch.full((3,), float(log_std_init)))

    def proba_distribution(self, mean_actions, log_std):
        if mean_actions.shape[-1] != 6 or log_std.shape != (3,):
            raise ValueError('six head outputs and three protection deviations required')
        logits = mean_actions[..., :3]
        self.direction = Categorical(logits=logits if self.cash_enabled else logits[..., [0, 2]])
        means = mean_actions[..., 3:]
        self.distribution = Normal(means, torch.ones_like(means) * log_std.exp())
        return self

    def _join(self, category, protections):
        direction = category - 1 if self.cash_enabled else 2 * category - 1
        return torch.cat((direction.to(protections.dtype).unsqueeze(-1), protections), dim=-1)

    def sample(self):
        return self._join(self.direction.sample(), self.distribution.rsample())

    def mode(self):
        return self._join(self.direction.probs.argmax(dim=-1), self.distribution.mean)

    def log_prob(self, actions):
        direction = actions[..., 0]
        valid = (direction == -1) | (direction == 1)
        if self.cash_enabled:
            valid = valid | (direction == 0)
        if actions.shape[-1] != 4 or not bool(valid.all()):
            raise ValueError('direction must be an available exact categorical action')
        category = direction + 1 if self.cash_enabled else (direction + 1) / 2
        return self.direction.log_prob(category.long()) + self.distribution.log_prob(actions[..., 1:]).sum(-1)

    def entropy(self):
        return self.direction.entropy() + self.distribution.entropy().sum(-1)


class HybridRecurrentPolicy(RecurrentActorCriticPolicy):
    def __init__(self, *args, cash_enabled, **kwargs):
        if type(cash_enabled) is not bool:
            raise ValueError('explicit boolean cash availability required')
        if kwargs.get('use_sde', False) or kwargs.get('squash_output', False):
            raise ValueError('hybrid policy requires unsquashed independent protections')
        self.cash_enabled = cash_enabled
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        if (self.action_space.shape != (4,) or not np.all(self.action_space.low == -1)
                or not np.all(self.action_space.high == 1)):
            raise ValueError('hybrid policy requires normalized four-component Box')
        self.action_dist = DirectionProtectionDistribution(self.cash_enabled)
        super()._build(lr_schedule)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        # sb3-contrib zero-pads recurrent minibatches before evaluating them,
        # then excludes padded rows with its rollout mask. Zero is outside the
        # binary support. Give only all-zero placeholder rows a valid dummy
        # direction so their unused likelihood/gradient remains finite. Real
        # sampled actions and the caller's tensor are unchanged; distribution
        # log_prob and execution validation still reject real binary cash.
        if not self.cash_enabled:
            padding = (actions == 0).all(dim=-1)
            if bool(padding.any()):
                actions = actions.clone()
                actions[padding, 0] = -1
        return super().evaluate_actions(obs, actions, lstm_states, episode_starts)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data['cash_enabled'] = self.cash_enabled
        return data

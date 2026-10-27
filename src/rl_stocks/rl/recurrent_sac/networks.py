"""Independent observation-memory networks using SB3's native action density."""
from copy import deepcopy

import torch
from torch import nn
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

from rl_stocks.training.recurrent_replay import _integer


class RecurrentActor(nn.Module):
    def __init__(self, observation_size: int, hidden_size: int):
        super().__init__()
        self.memory = nn.GRU(observation_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.mean = nn.Linear(hidden_size, 4)
        self.log_std = nn.Linear(hidden_size, 4)

    def sample(self, features, noise=None):
        latent = self.head(features)
        mean = self.mean(latent)
        log_std = self.log_std(latent).clamp(-20, 2)
        # SB3 sums log-prob over dimension1: flatten batch/time before calling it.
        distribution = SquashedDiagGaussianDistribution(4)
        distribution.proba_distribution(mean.reshape(-1, 4), log_std.reshape(-1, 4))
        if noise is None:
            gaussian = distribution.distribution.rsample()
        else:
            if (noise.shape != mean.shape or noise.dtype != mean.dtype or noise.device != mean.device
                    or not bool(torch.isfinite(noise).all())):
                raise ValueError('finite matching Gaussian noise required')
            gaussian = (mean + log_std.exp() * noise).reshape(-1, 4)
        actions = torch.tanh(gaussian)
        log_prob = distribution.log_prob(actions, gaussian)
        return actions.reshape(mean.shape), log_prob.reshape(mean.shape[:-1])


class RecurrentCritic(nn.Module):
    def __init__(self, observation_size: int, hidden_size: int):
        super().__init__()
        self.memory = nn.GRU(observation_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size + 4, hidden_size), nn.ReLU(),
                                  nn.Linear(hidden_size, 1))

    def forward(self, features, actions):
        return self.head(torch.cat((features, actions), dim=-1)).squeeze(-1)


class RecurrentSACNetworks(nn.Module):
    def __init__(self, observation_size: int, hidden_size: int = 64):
        super().__init__()
        _integer('observation_size', observation_size, 1)
        _integer('hidden_size', hidden_size, 1)
        self.observation_size = observation_size
        self.hidden_size = hidden_size
        self.actor = RecurrentActor(observation_size, hidden_size)
        self.critics = nn.ModuleList([RecurrentCritic(observation_size, hidden_size) for _ in range(2)])
        self.targets = deepcopy(self.critics)
        self.targets.requires_grad_(False)

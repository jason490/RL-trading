"""Recurrent PPO with a tanh-transformed diagonal Gaussian action distribution."""
import numpy as np
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution


class SquashedRecurrentPolicy(RecurrentActorCriticPolicy):
    def __init__(self,*args,use_sde=False,squash_output=True,**kwargs):
        if use_sde is not False or squash_output is not True:
            raise ValueError('fixed squashed Gaussian requires squash_output=True and use_sde=False')
        # The base constructor only exposes squashing for its own gSDE distribution.
        # Our _build supplies the independent squashed distribution instead, then
        # marks its output for the inherited prediction/unscale implementation.
        super().__init__(*args,use_sde=False,squash_output=False,**kwargs)

    def _build(self,lr_schedule):
        if (self.action_space.shape!=(4,) or not np.all(self.action_space.low==-1)
                or not np.all(self.action_space.high==1)):
            raise ValueError('squashed recurrent policy requires normalized four-component Box')
        self.action_dist=SquashedDiagGaussianDistribution(4,epsilon=1e-6)
        super()._build(lr_schedule)
        self._squash_output=True

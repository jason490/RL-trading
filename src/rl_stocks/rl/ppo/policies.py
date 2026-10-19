"""SB3 PPO actor branches that optionally isolate exposure from account inputs."""
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor


def network(inputs,widths):
    return nn.Sequential(nn.Linear(inputs,widths[0]),nn.Tanh(),
                         nn.Linear(widths[0],widths[1]),nn.Tanh())


class BranchExtractor(nn.Module):
    def __init__(self,features,market_features,widths):
        super().__init__()
        self.market_features=market_features;self.branch_width=widths[-1]
        self.latent_dim_pi=2*self.branch_width;self.latent_dim_vf=self.branch_width
        self.market_actor=network(market_features,widths)
        self.full_actor=network(features,widths)
        self.value=network(features,widths)
    def forward_actor(self,features):
        return torch.cat((self.market_actor(features[...,:self.market_features]),
                          self.full_actor(features)),dim=-1)
    def forward_critic(self,features):return self.value(features)
    def forward(self,features):return self.forward_actor(features),self.forward_critic(features)


class ConnectedActionHead(nn.Module):
    """Wrap the existing initialized linear layer without new parameters or RNG."""
    def __init__(self,linear,market_width,market_only):
        super().__init__();self.linear=linear
        self.register_buffer('connection_mask',torch.ones_like(linear.weight))
        if market_only:self.connection_mask[0,market_width:]=0
    @property
    def bias(self):return self.linear.bias
    @property
    def weight(self):return self.linear.weight
    def forward(self,features):return F.linear(features,self.weight*self.connection_mask,self.bias)


class BranchPolicy(ActorCriticPolicy):
    def __init__(self,*args,market_features,direction_market_only=False,**kwargs):
        if type(market_features) is not int or market_features<1 or type(direction_market_only) is not bool:
            raise ValueError('invalid branch policy information contract')
        if kwargs.get('use_sde',False) or kwargs.get('features_extractor_class',FlattenExtractor)!=FlattenExtractor:
            raise ValueError('branch policy requires diagonal Gaussian and raw flattened features')
        self.market_features=market_features;self.direction_market_only=direction_market_only
        super().__init__(*args,**kwargs)
    def _build_mlp_extractor(self):
        if (len(self.observation_space.shape)!=1 or not 0<self.market_features<self.features_dim
            or self.action_space.shape!=(4,) or not isinstance(self.net_arch,(list,tuple))
            or len(self.net_arch)!=2 or any(type(w) is not int or w<1 for w in self.net_arch)
            or self.activation_fn is not nn.Tanh):
            raise ValueError('branch policy requires vector market/account inputs, four actions and two Tanh layers')
        self.mlp_extractor=BranchExtractor(self.features_dim,self.market_features,self.net_arch).to(self.device)
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        # SB3's optimizer still owns the same Parameter objects. Only their
        # module path changes; the fixed buffer never enters the optimizer.
        self.action_net=ConnectedActionHead(self.action_net,self.mlp_extractor.branch_width,self.direction_market_only)
    def _get_constructor_parameters(self):
        data=super()._get_constructor_parameters()
        data.update(market_features=self.market_features,direction_market_only=self.direction_market_only)
        return data

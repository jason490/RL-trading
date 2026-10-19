import importlib
import math
import numpy as np
from stable_baselines3 import PPO,SAC,TD3
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.noise import NormalActionNoise

CLASSES={"ppo":PPO,"sac":SAC,"td3":TD3,"recurrent_ppo":RecurrentPPO}
ON_POLICY=('ppo','recurrent_ppo')

def resolved_parameters(algorithm,smoke=False,overrides=None,policy_variant="native",market_features=None):
    if algorithm not in CLASSES:raise ValueError("unsupported algorithm")
    from rl_stocks.rl.ppo.architecture import architecture_spec
    architecture_spec(policy_variant,market_features)
    if policy_variant!='native' and algorithm!='ppo':raise ValueError('branch architecture requires PPO')
    params=importlib.import_module(f"rl_stocks.rl.{algorithm}.config").parameters(smoke)
    params.update(learning_rate=3e-4,gamma=.99,policy_kwargs={"net_arch":[32,32] if smoke else [64,64]})
    if algorithm=='recurrent_ppo':
        params['policy_kwargs'].update(lstm_hidden_size=32 if smoke else 64,n_lstm_layers=1,
                                      shared_lstm=False,enable_critic_lstm=True)
    params.update(overrides or {})
    if policy_variant!='native':
        if any(k in params['policy_kwargs'] for k in ('market_features','direction_market_only','features_extractor_class','use_sde')):
            raise ValueError('branch information contract cannot be overridden')
        params['policy_kwargs']={**params['policy_kwargs'],'market_features':market_features,
            'direction_market_only':policy_variant=='two_branch_market_exposure'}
    if not isinstance(params["batch_size"],int) or params["batch_size"]<=0:
        raise ValueError("batch size must be a positive integer")
    if not math.isfinite(params["learning_rate"]) or params["learning_rate"]<=0 or not 0<params["gamma"]<=1:
        raise ValueError("invalid learning rate/discount")
    if algorithm in ON_POLICY and (params["n_steps"]<2 or params["n_steps"]%params["batch_size"]):
        raise ValueError("PPO rollout must be divisible by batch size")
    if algorithm not in ON_POLICY and params["learning_starts"]<0:raise ValueError("invalid warmup")
    if algorithm not in ON_POLICY and params["buffer_size"]<params["batch_size"]:
        raise ValueError("replay capacity must cover the batch")
    return params

def create(algorithm,env,seed=11,device="cpu",smoke=False,overrides=None,policy_variant="native"):
    market_features=len(env.names) if policy_variant!='native' else None
    if policy_variant!='native' and getattr(env,'observation_profile',None)!='execution_full':
        raise ValueError('branch study requires full execution observation profile')
    params=resolved_parameters(algorithm,smoke,overrides,policy_variant,market_features)
    if algorithm=="td3":params["action_noise"]=NormalActionNoise(np.zeros(4),np.full(4,.1))
    policy='MlpLstmPolicy' if algorithm=='recurrent_ppo' else 'MlpPolicy'
    if policy_variant!='native':
        from rl_stocks.rl.ppo.policies import BranchPolicy
        policy=BranchPolicy
    return CLASSES[algorithm](policy,env,seed=seed,device=device,verbose=0,**params)

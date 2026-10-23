"""Exact conditional boundary mass; no action sampling, fitting or return scoring."""
import math
import numpy as np

COMPONENTS=('exposure','stop_distance','target_near','zone_width')


def boundary_mass(means,stds):
    means=np.asarray(means,dtype=np.float64);stds=np.asarray(stds,dtype=np.float64)
    if (means.ndim!=2 or means.shape[1]!=4 or not means.shape[0] or stds.shape!=means.shape
            or not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds<=0).any()):
        raise ValueError('nonempty finite Nx4 Gaussian means and positive matching standard deviations required')
    lower=[];upper=[];joint=[]
    for mus,sigmas in zip(means,stds,strict=True):
        lo=[.5*math.erfc((1.+float(mu))/float(sd)/math.sqrt(2.)) for mu,sd in zip(mus,sigmas,strict=True)]
        hi=[.5*math.erfc((1.-float(mu))/float(sd)/math.sqrt(2.)) for mu,sd in zip(mus,sigmas,strict=True)]
        total=[min(1.,a+b) for a,b in zip(lo,hi,strict=True)]
        joint.append(1. if any(p==1. for p in total) else -math.expm1(sum(math.log1p(-p) for p in total)))
        lower.append(lo);upper.append(hi)
    lower=np.asarray(lower);upper=np.asarray(upper)
    return dict(observations=len(means),components={name:dict(mean=float(means[:,i].mean()),
        mean_min=float(means[:,i].min()),mean_max=float(means[:,i].max()),
        std_mean=float(stds[:,i].mean()),std_min=float(stds[:,i].min()),std_max=float(stds[:,i].max()),
        below_lower_probability=float(lower[:,i].mean()),above_upper_probability=float(upper[:,i].mean()),
        outside_probability=float((lower[:,i]+upper[:,i]).mean())) for i,name in enumerate(COMPONENTS)},
        any_component_outside_probability=float(np.mean(joint)),
        interpretation='Conditional Gaussian probability on fixed observations, not observed training-rollout clipping frequency')


def recurrent_parameters(model,episodes,normalizer,names,decisions=32):
    """Carry actor memory within fixed flat-account prefixes; never execute policy outputs."""
    import torch
    from stable_baselines3.common.distributions import DiagGaussianDistribution
    from rl_stocks.environment.trading.env import TradingEnv
    if (type(decisions) is not int or decisions<=0 or not episodes
            or any(e['split']!='train' or len(e['bars'])<=decisions for e in episodes)
            or type(model.policy.action_dist) is not DiagGaussianDistribution):
        raise ValueError('training prefixes and native diagonal Gaussian recurrent policy required')
    policy=model.policy;lstm=policy.lstm_actor;means=[];stds=[];symbols=[]
    was_training=policy.training
    policy.set_training_mode(False)
    try:
        for e in sorted(episodes,key=lambda e:e['symbol']):
            env=TradingEnv([e],normalizer,names,observation_profile='execution_masked')
            try:
                obs,_=env.reset(seed=11);state=tuple(torch.zeros((lstm.num_layers,1,lstm.hidden_size),device=model.device) for _ in range(2))
                for i in range(decisions):
                    with torch.no_grad():
                        dist,state=policy.get_distribution(torch.as_tensor(obs,device=model.device).reshape(1,-1),state,
                            torch.tensor([i==0],dtype=torch.float32,device=model.device))
                        means.append(dist.distribution.mean.cpu().numpy()[0].copy())
                        stds.append(dist.distribution.stddev.cpu().numpy()[0].copy())
                    obs,_,terminated,truncated,_=env.step(np.zeros(4,dtype=np.float32))
                    if terminated or truncated:raise ValueError('complete fixed prefix required')
                    if env.sim.portfolio.quantity!=0:raise ValueError('flat-account diagnostic invariant violated')
                symbols.append(e['symbol'])
            finally:env.close()
    finally:policy.set_training_mode(was_training)
    return dict(symbols=symbols,decisions_per_symbol=decisions,**boundary_mass(means,stds))

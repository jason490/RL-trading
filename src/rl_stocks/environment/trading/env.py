from dataclasses import asdict
import gymnasium as gym
import numpy as np
from rl_stocks.core.contracts import Bar
from rl_stocks.environment.execution.simulator import Simulator, SimulationConfig
from rl_stocks.environment.rewards.equity import equity_reward

class TradingEnv(gym.Env):
    metadata={"render_modes":[]}
    def __init__(self,episodes,normalizer,names,config=SimulationConfig(),random_start=False,episode_steps=None,reward_scale=1.):
        super().__init__()
        if not episodes:raise ValueError("no episodes")
        if tuple(names)!=normalizer.names:raise ValueError("feature names mismatch")
        if len({e["split"] for e in episodes})!=1:raise ValueError("split mixing")
        self.episodes=episodes;self.normalizer=normalizer;self.names=names;self.config=config
        self.random_start=random_start;self.episode_steps=episode_steps
        if not np.isfinite(reward_scale) or reward_scale<=0:raise ValueError('positive finite reward scale required')
        self.reward_scale=float(reward_scale)
        self.action_space=gym.spaces.Box(-1.,1.,shape=(4,),dtype=np.float32)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(len(names)+5,),dtype=np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        index=(options or {}).get("episode",int(self.np_random.integers(len(self.episodes))))
        self.episode=self.episodes[index]
        self.bars=[Bar(**b) for b in self.episode["bars"]]
        self.i=int(self.np_random.integers(max(1,len(self.bars)-self.config.max_holding-1))) if self.random_start else 0
        self.steps=0;self.done=False;self.sim=Simulator(self.config)
        return self.observation(),{"symbol":self.episode["symbol"],"session":self.bars[self.i].session}
    def observation(self):
        p=self.sim.portfolio;price=self.bars[self.i].close;equity=p.equity(price)
        vector=self.normalizer.transform(self.episode["features"][self.i],self.names)
        state=[p.quantity*price/max(equity,1e-8),p.cash/max(equity,1e-8),
            p.age/self.config.max_holding,(price-p.basis)*p.quantity/max(equity,1e-8),
            equity/self.config.initial_equity-1]
        value=np.asarray(vector+state,dtype=np.float32)
        if not np.isfinite(value).all():raise ValueError("nonfinite observation")
        return value
    def step(self,action):
        if self.done:raise RuntimeError("reset required after episode end")
        previous=self.bars[self.i];atr=self.episode["atr"][self.i]
        self.i+=1;self.steps+=1
        bar=self.bars[self.i]
        event_start=len(self.sim.portfolio.events)
        info=self.sim.advance(action,previous,bar,atr)
        terminated=info["equity"]<=0
        truncated=not terminated and (self.i==len(self.bars)-1 or
            (self.episode_steps is not None and self.steps>=self.episode_steps))
        # Data cutoffs/time limits use marked equity and bootstrap; no artificial
        # liquidation fee is added at truncation. Episodes never cross split ends.
        self.done=terminated or truncated
        reward=equity_reward(info["previous_equity"],info["equity"])
        info.update(session=bar.session,price=bar.close,events=self.sim.portfolio.events[event_start:])
        info['unscaled_reward']=reward
        return self.observation(),float(self.reward_scale*reward),bool(terminated),bool(truncated),info

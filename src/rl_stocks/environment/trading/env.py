import gymnasium as gym
import numpy as np
from rl_stocks.core.contracts import Bar
from rl_stocks.environment.execution.simulator import Simulator, SimulationConfig
from rl_stocks.environment.rewards.equity import equity_reward, validate_reward_kind

from rl_stocks.environment.observations.state import observation_spec, portfolio_state

class TradingEnv(gym.Env):
    metadata={"render_modes":[]}
    def __init__(self,episodes,normalizer,names,config=SimulationConfig(),random_start=False,episode_steps=None,reward_scale=1.,start_sampling='legacy_uniform',observation_profile='legacy',reward_kind='net_log_equity',exposure_execution='unrestricted'):
        super().__init__()
        if not episodes:raise ValueError("no episodes")
        if tuple(names)!=normalizer.names:raise ValueError("feature names mismatch")
        if len({e["split"] for e in episodes})!=1:raise ValueError("split mixing")
        self.episodes=episodes;self.normalizer=normalizer;self.names=names;self.config=config
        self.random_start=random_start;self.episode_steps=episode_steps
        from rl_stocks.environment.sampling.regimes import MODES,CausalStarts
        if start_sampling not in MODES:raise ValueError('unknown start sampling mode')
        if start_sampling!='legacy_uniform' and (not random_start or episodes[0]['split']!='train'):
            raise ValueError('causal start sampling requires random-start training only')
        if start_sampling=='stock_etf_cycle_windows':
            from rl_stocks.environment.sampling.windows import StockEtfCycles
            self.start_sampler=StockEtfCycles(episodes,episode_steps)
        elif start_sampling=='full_window_uniform':
            from rl_stocks.environment.sampling.windows import FullWindows
            self.start_sampler=FullWindows(episodes,episode_steps)
        else:
            self.start_sampler=CausalStarts(episodes,start_sampling,config.max_holding) if start_sampling!='legacy_uniform' else None
        self.reset_sampling_statistics()
        if not np.isfinite(reward_scale) or reward_scale<=0:raise ValueError('positive finite reward scale required')
        self.reward_scale=float(reward_scale)
        self.reward_kind=validate_reward_kind(reward_kind)
        from rl_stocks.risk.exposure import execution_contract
        execution_contract(exposure_execution,exposure_execution)
        self.exposure_execution=exposure_execution
        self.observation_profile=observation_profile
        self.observation_spec=observation_spec(observation_profile)
        self.action_space=gym.spaces.Box(-1.,1.,shape=(4,),dtype=np.float32)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(len(names)+len(self.observation_spec["portfolio_fields"]),),dtype=np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None and self.start_sampler and self.start_sampler.mode=='stock_etf_cycle_windows':
            self.start_sampler.reset_cycles()
        if self.start_sampler is None:
            index=(options or {}).get("episode",int(self.np_random.integers(len(self.episodes))))
            start=None
        else:
            if 'episode' in (options or {}):raise ValueError('explicit episode conflicts with sampled training pool')
            index,start=self.start_sampler.choose(self.np_random)
        self.episode_index=index
        self.episode=self.episodes[index]
        self.bars=[Bar(**b) for b in self.episode["bars"]]
        self.i=start if start is not None else int(self.np_random.integers(max(1,len(self.bars)-self.config.max_holding-1))) if self.random_start else 0
        if self.start_sampler:
            self.sampling_counts['starts']+=1
            self.sampling_counts['downtrend_starts']+=int(self.start_sampler.label(index,self.i))
        if self.start_sampler and self.start_sampler.mode in ('full_window_uniform','stock_etf_cycle_windows'):
            group=self.start_sampler.groups[index]
            self.sampling_counts['history_groups'].setdefault(group,dict(starts=0,transitions=0))['starts']+=1
        self.steps=0;self.done=False;self.sim=Simulator(self.config)
        return self.observation(),{"symbol":self.episode["symbol"],"session":self.bars[self.i].session}
    def reset_sampling_statistics(self):
        if self.start_sampler and self.start_sampler.mode=='stock_etf_cycle_windows':self.start_sampler.reset_cycles()
        self.sampling_counts=dict(starts=0,downtrend_starts=0,transitions=0,downtrend_transitions=0)
        if self.start_sampler and self.start_sampler.mode in ('full_window_uniform','stock_etf_cycle_windows'):
            self.sampling_counts['history_groups']={}
            self.sampling_counts['downtrend_counts_measured']=False
    def sampling_statistics(self):
        from copy import deepcopy
        return {**deepcopy(self.sampling_counts),'mode':self.start_sampler.mode} if self.start_sampler else None
    def observation(self):
        p=self.sim.portfolio;price=self.bars[self.i].close;equity=p.equity(price)
        vector=self.normalizer.transform(self.episode["features"][self.i],self.names)
        state=portfolio_state(self.sim,price,self.episode["atr"][self.i],self.observation_profile)
        value=np.asarray(vector+state,dtype=np.float32)
        if not np.isfinite(value).all():raise ValueError("nonfinite observation")
        return value
    def step(self,action):
        if self.done:raise RuntimeError("reset required after episode end")
        if self.start_sampler:
            self.sampling_counts['transitions']+=1
            self.sampling_counts['downtrend_transitions']+=int(self.start_sampler.label(self.episode_index,self.i))
        if self.start_sampler and self.start_sampler.mode in ('full_window_uniform','stock_etf_cycle_windows'):
            group=self.start_sampler.groups[self.episode_index]
            self.sampling_counts['history_groups'].setdefault(group,dict(starts=0,transitions=0))['transitions']+=1
        previous=self.bars[self.i];atr=self.episode["atr"][self.i]
        self.i+=1;self.steps+=1
        bar=self.bars[self.i]
        event_start=len(self.sim.portfolio.events)
        if self.exposure_execution!='unrestricted':
            from rl_stocks.risk.exposure import filter_exposure
            original_action=tuple(float(x) for x in action)
            action=filter_exposure(action,self.exposure_execution)
        info=self.sim.advance(action,previous,bar,atr)
        if self.exposure_execution!='unrestricted':
            info.update(policy_action=original_action,exposure_execution=self.exposure_execution)
        terminated=info["equity"]<=0
        truncated=not terminated and (self.i==len(self.bars)-1 or
            (self.episode_steps is not None and self.steps>=self.episode_steps))
        # Data cutoffs/time limits use marked equity and bootstrap; no artificial
        # liquidation fee is added at truncation. Episodes never cross split ends.
        self.done=terminated or truncated
        reward=equity_reward(info["previous_equity"],info["equity"],self.reward_kind,self.config.initial_equity)
        info.update(session=bar.session,price=bar.close,events=self.sim.portfolio.events[event_start:])
        info['unscaled_reward']=reward
        info['reward_kind']=self.reward_kind
        return self.observation(),float(self.reward_scale*reward),bool(terminated),bool(truncated),info

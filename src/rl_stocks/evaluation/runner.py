from dataclasses import asdict
from rl_stocks.core.contracts import Bar
from rl_stocks.environment.accounting.portfolio import Portfolio
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.evaluation.metrics import metrics
from rl_stocks.evaluation.direction import directional_attribution
from rl_stocks.rl.registry.prediction import PredictionStream

def evaluate(model,dataset,split="validation",config=SimulationConfig(),observation_profile="legacy",exposure_mode="unrestricted"):
    from rl_stocks.environment.trading.env import TradingEnv
    if split not in ("validation","test"):raise ValueError("evaluation split required")
    return _evaluate(model,dataset,split,config,observation_profile,exposure_mode)


def _evaluate(model,dataset,split,config,observation_profile,exposure_mode="unrestricted"):
    from rl_stocks.environment.trading.env import TradingEnv
    from rl_stocks.evaluation.exposure import EXPOSURE_MODES,filter_exposure
    if exposure_mode not in EXPOSURE_MODES:raise ValueError('unknown exposure filter')
    normalizer=normalizer_from(dataset["normalizer"])
    results={}
    for key,episode in dataset["episodes"].items():
        if episode["split"]!=split:continue
        env=TradingEnv([episode],normalizer,dataset["names"],config,observation_profile=observation_profile)
        obs,_=env.reset(seed=11,options={"episode":0})
        rows=[];prediction=PredictionStream(model)
        while True:
            action=prediction.predict(obs)
            policy_action=tuple(float(x) for x in action)
            if exposure_mode!="unrestricted":action=filter_exposure(action,exposure_mode)
            obs,reward,terminated,truncated,info=env.step(action)
            info.pop("events",None)
            if exposure_mode!="unrestricted":info.update(policy_action=policy_action,exposure_mode=exposure_mode)
            rows.append(info)
            if terminated or truncated:break
        events=env.sim.portfolio.events
        results[key]={"rows":rows,"events":events,"metrics":metrics(rows,events,config.initial_equity),
            "baselines":baselines(episode,config),"split":split,"symbol":episode["symbol"],
            "directional":directional_attribution(episode,rows,events,config.initial_equity)}
    if not results:raise ValueError("no evaluation episodes")
    return results

def baselines(episode,config):
    bars=[Bar(**b) for b in episode["bars"]]
    p=Portfolio(cash=config.initial_equity)
    rows=[]
    for i,b in enumerate(bars[1:]):
        p.corporate_action(b)
        if i==0:
            shares=int(p.cash/(b.open*(1+(config.fee_bps+config.impact_bps)/10000)))
            p.trade(shares,b.open,b.open_time,"buy_hold",config.fee_bps,config.impact_bps)
        p.age+=1
        rows.append(dict(session=b.session,equity=p.equity(b.close),costs=p.costs,turnover=p.turnover))
    return {"cash":{"metrics":metrics([dict(equity=config.initial_equity)],[],config.initial_equity)},
            "buy_hold":{"rows":rows,"metrics":metrics(rows,p.events,config.initial_equity)}}

def selection_score(results):
    # Equal-weight symbol/episode average net return, declared before testing.
    return sum(r["metrics"]["net_return"] for r in results.values())/len(results)

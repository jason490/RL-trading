"""Independent recorded-action economics and optional local frozen GRU parity."""
import numpy as np
import torch

from rl_stocks.core.contracts import canonical
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.evaluation.report_contract import validate_development_report
from rl_stocks.rl.recurrent_sac.prediction import FrozenRecurrentActor


def verify_report(results, dataset, networks=None, *, observation_profile='execution_full'):
    """Replay recorded actions, keeping CPU actor arithmetic separate from economics."""
    if networks is not None and observation_profile!='execution_full':
        raise ValueError('GRU frozen parity requires full execution observations')
    from rl_stocks.environment.execution.simulator import SimulationConfig
    validate_development_report(results, dataset, SimulationConfig())
    normalizer = normalizer_from(dataset['normalizer'])
    for key, result in results.items():
        episode = dataset['episodes'][key]
        env = TradingEnv([episode], normalizer, dataset['names'], observation_profile=observation_profile)
        try:
            obs, _ = env.reset(seed=11, options={'episode': 0})
            frozen = None if networks is None else FrozenRecurrentActor(networks, observation_schema='execution-full-v1-normalized21')
            hidden = state = None
            for i, row in enumerate(result['rows']):
                if networks is not None:
                    with torch.no_grad():
                        features, hidden = networks.actor.memory(torch.tensor(obs).reshape(1,1,-1), hidden)
                        direct = networks.actor.mean(networks.actor.head(features)).tanh().reshape(4).numpy()
                    inferred, state = frozen.predict(obs, observation_schema=frozen.observation_schema,
                                                     state=state, episode_start=i == 0)
                    if not np.array_equal(direct, inferred):
                        raise ValueError('local frozen carried/reset prediction parity failed')
                action = np.asarray(row['raw_action'], dtype=np.float32)
                if (action.shape != (4,) or not np.isfinite(action).all() or (np.abs(action)>1).any()
                        or abs(float(action[0])) not in (0, 1) or row.get('exposure_mode') != 'unit_magnitude'
                        or np.sign(row['policy_action'][0]) != action[0]
                        or tuple(row['policy_action'][1:]) != tuple(row['raw_action'][1:])):
                    raise ValueError('recorded exposure/protective action contract mismatch')
                obs, _, _, _, actual = env.step(action)
                for name in ('equity','cash','quantity','costs','turnover','price','session','age',
                             'borrow_cost','raw_return','previous_equity'):
                    if canonical(actual[name]) != canonical(row[name]):
                        raise ValueError('saved development economics differ from recorded-action replay')
            if canonical(env.sim.portfolio.events) != canonical(result['events']):
                raise ValueError('saved fills differ from recorded-action replay')
        finally:
            env.close()


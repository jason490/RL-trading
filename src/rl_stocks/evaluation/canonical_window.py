"""Reference actor replay using the shared environment without an RL dependency."""
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.evaluation.canonical_task import CanonicalEpisodePredictor
from rl_stocks.evaluation.direction import directional_attribution
from rl_stocks.evaluation.metrics import metrics
from rl_stocks.evaluation.runner import baselines
from rl_stocks.risk.exposure import filter_exposure


def replay_window(frozen, view):
    """Same accounting/report contract as _evaluate, with no PredictionStream."""
    normalizer = normalizer_from(view['normalizer'])
    config = SimulationConfig()
    mode = frozen.execution['evaluation_exposure_mode']
    results = {}
    for key, episode in view['episodes'].items():
        predictor = CanonicalEpisodePredictor(frozen, episode)
        env = TradingEnv([episode], normalizer, view['names'], config,
                         observation_profile='execution_full')
        try:
            observation, _ = env.reset(seed=11, options={'episode': 0})
            rows = []
            while True:
                action, _ = predictor.predict(observation, deterministic=True)
                policy_action = tuple(float(x) for x in action)
                approved = filter_exposure(action, mode)
                observation, _, terminated, truncated, info = env.step(approved)
                info.pop('events', None)
                if mode != 'unrestricted':
                    info.update(policy_action=policy_action, exposure_mode=mode)
                rows.append(info)
                if terminated or truncated:
                    break
            if predictor.cursor != len(episode['bars']) - 1:
                raise ValueError('incomplete fixed reference episode replay')
            events = env.sim.portfolio.events
            results[key] = {'rows': rows, 'events': events,
                'metrics': metrics(rows, events, config.initial_equity),
                'baselines': baselines(episode, config), 'split': 'train', 'symbol': episode['symbol'],
                'directional': directional_attribution(episode, rows, events, config.initial_equity)}
        finally:
            env.close()
    if not results:
        raise ValueError('reference window cannot be empty')
    return results

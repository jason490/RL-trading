"""Development-only recurrent SAC adapter over the common simulator/report path."""
from datetime import date

from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.runner import evaluate
from rl_stocks.rl.recurrent_sac.networks import RecurrentSACNetworks
from rl_stocks.rl.recurrent_sac.prediction import FrozenRecurrentActor


def evaluate_recurrent_sac(networks, dataset, *, exposure_mode: str,
                           split='validation', config=SimulationConfig()):
    # No generic test route: the retired 2024+ interval cannot enter new research.
    if split != 'validation':
        raise ValueError('recurrent SAC development evaluation permits validation only')
    if not isinstance(networks, RecurrentSACNetworks) or networks.observation_size != 21:
        raise ValueError('execution-full21 recurrent SAC networks required')
    if dataset.get('schema') != 'daily-indicators-v1' or len(dataset.get('names', [])) != 7:
        raise ValueError('fixed seven-field baseline feature schema required')
    normalizer = normalizer_from(dataset['normalizer'])
    if tuple(dataset['names']) != normalizer.names:
        raise ValueError('frozen normalizer feature order mismatch')
    development = []
    for ep in dataset['episodes'].values():
        if ep['split'] not in ('train', 'validation'):
            raise ValueError('retired/test episodes must be excluded before loading research payload')
        if ep['split'] == 'validation':
            development.append(ep)
            sessions = [date.fromisoformat(bar['session']) for bar in ep['bars']]
            if (len(sessions) < 2 or any(not date(2021, 1, 1) <= d < date(2024, 1, 1) for d in sessions)
                    or any(b <= a for a, b in zip(sessions, sessions[1:]))):
                raise ValueError('chronological 2021-2023 development bars required')
    if not development:
        raise ValueError('development episodes required')
    frozen = FrozenRecurrentActor(networks, observation_schema='execution-full-v1-normalized21')
    return evaluate(frozen, dataset, split='validation', config=config,
                     observation_profile='execution_full', exposure_mode=exposure_mode)

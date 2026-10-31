"""Validated full-state task environments with unchanged causal group sampling."""
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.environment.observations.task import InstrumentTypeObservation
from rl_stocks.environment.trading.env import TradingEnv
from rl_stocks.training.task_conditioning import validate_type_manifest


def build_task_environment(dataset, type_manifest, *, task_role, smoke=False):
    if type(smoke) is not bool:
        raise ValueError('explicit boolean smoke flag required')
    validate_type_manifest(dataset, type_manifest)
    episodes = [e for e in dataset['episodes'].values() if e['split'] == 'train']
    normalizer = normalizer_from(dataset['normalizer'])
    if len(dataset['names']) != 7:
        raise ValueError('task comparison requires seven market features')
    environment = TradingEnv(episodes, normalizer, dataset['names'],
                             random_start=True, episode_steps=128 if smoke else 512,
                             start_sampling='stock_etf_cycle_windows',
                             observation_profile='execution_full',
                             reward_kind='net_log_equity', reward_scale=1.,
                             exposure_execution='unrestricted')
    try:
        return InstrumentTypeObservation(environment, type_manifest['records'], task_role=task_role)
    except BaseException:
        environment.close()
        raise

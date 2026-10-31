"""Task SAC training-window replay through the unchanged accounting evaluator."""
from datetime import date

import numpy as np
from stable_baselines3 import SAC

from rl_stocks.core.contracts import digest
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.runner import _evaluate
from rl_stocks.evaluation.training_fit import training_window
from rl_stocks.rl.sac.task_policy import TaskSACPolicy
from rl_stocks.training.task_conditioning import validate_type_manifest


class EpisodeTaskPredictor:
    """One fixed chronological stream; never share a cursor between episodes."""
    def __init__(self, model, episode, record):
        if (not isinstance(model, SAC) or type(model.policy) is not TaskSACPolicy
                or model.observation_space.shape != (22,)):
            raise ValueError('task SAC model required')
        if episode['symbol'] != record['symbol']:
            raise ValueError('episode type identity mismatch')
        if record['kind'] not in ('stock', 'etf'):
            raise ValueError('unknown instrument type')
        self.model = model
        self.role = model.policy.task_role
        self.tag = 0 if self.role == 'masked_shared' else (1 if record['kind'] == 'etf' else -1)
        self.sessions = tuple(bar['session'] for bar in episode['bars'][:-1])
        self.known = date.fromisoformat(record['known_from'])
        self.cursor = 0

    def predict(self, observation, deterministic=True):
        if deterministic is not True:
            raise ValueError('task evaluation must be deterministic')
        if self.cursor >= len(self.sessions):
            raise ValueError('task episode prediction stream exhausted')
        if date.fromisoformat(self.sessions[self.cursor]) < self.known:
            raise ValueError('historical type unavailable at decision')
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (21,) or not np.isfinite(observation).all():
            raise ValueError('task replay requires finite full21 input')
        vector = np.concatenate((observation, np.asarray([self.tag], dtype=np.float32)))
        action, state = self.model.predict(vector, deterministic=True)
        self.cursor += 1
        return action, state


def evaluate_task_training(model, dataset, type_manifest):
    """Fixed original26 fitted window only; this API cannot open development/test."""
    validate_type_manifest(dataset, type_manifest)
    view = training_window(dataset, original_panel_only=True)
    records = {r['symbol']: r for r in type_manifest['records']}
    protected = digest((dataset['normalizer'], type_manifest))
    results = {}
    try:
        for key, episode in view['episodes'].items():
            adapter = EpisodeTaskPredictor(model, episode, records[episode['symbol']])
            part = _evaluate(adapter, {**view, 'episodes': {key: episode}}, 'train',
                             SimulationConfig(), 'execution_full', 'unit_magnitude')
            if adapter.cursor != len(episode['bars']) - 1:
                raise ValueError('incomplete task training replay')
            results.update(part)
    finally:
        if digest((dataset['normalizer'], type_manifest)) != protected:
            raise ValueError('task preprocessing/provenance mutated')
    return results

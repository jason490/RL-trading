"""Reference task execution integration; training-only diagnostic access."""
from datetime import date

from rl_stocks.core.contracts import digest
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.runner import _evaluate
from rl_stocks.evaluation.training_fit import training_window
from rl_stocks.inference.canonical_bundle import FrozenCanonicalTaskPolicy
from rl_stocks.training.task_conditioning import validate_type_manifest


class CanonicalEpisodePredictor:
    """Raw deterministic actions; the evaluator applies execution filtering once."""
    def __init__(self, frozen, episode):
        if not isinstance(frozen, FrozenCanonicalTaskPolicy):
            raise ValueError('frozen reference policy required')
        sessions = tuple(b['session'] for b in episode['bars'])
        if (episode['split'] != 'train' or len(sessions) < 2
                or sessions != tuple(sorted(set(sessions)))
                or any(date.fromisoformat(s) > date(2020, 12, 31) for s in sessions)):
            raise ValueError('chronological training-only reference episode required')
        self.frozen, self.symbol = frozen, episode['symbol']
        self.sessions, self.cursor = sessions[:-1], 0

    def predict(self, observation, deterministic=True):
        if deterministic is not True:
            raise ValueError('reference evaluation must be deterministic')
        if self.cursor >= len(self.sessions):
            raise ValueError('reference episode prediction stream exhausted')
        action = self.frozen.raw_action(observation, symbol=self.symbol,
                                       session=self.sessions[self.cursor])
        self.cursor += 1
        return action, None


def evaluate_canonical_training(frozen, dataset, type_manifest):
    """Fixed original26 fitted window; no development/test or model selection."""
    validate_type_manifest(dataset, type_manifest)
    source = frozen.manifest['source_manifest']
    contract = source['metadata']['task_contract']
    if (digest(dataset) != contract['dataset_sha256']
            or digest(type_manifest) != contract['type_manifest_sha256']
            or dataset['schema'] != source['feature_schema']
            or digest(dataset['normalizer']) != digest(source['normalizer'])):
        raise ValueError('reference evaluation source/preprocessing mismatch')
    view = training_window(dataset, original_panel_only=True)
    protected = digest((dataset['normalizer'], type_manifest, frozen.manifest))
    results = {}
    try:
        for key, episode in view['episodes'].items():
            predictor = CanonicalEpisodePredictor(frozen, episode)
            results.update(_evaluate(predictor, {**view, 'episodes': {key: episode}}, 'train',
                                     SimulationConfig(), 'execution_full',
                                     frozen.execution['evaluation_exposure_mode']))
            if predictor.cursor != len(episode['bars']) - 1:
                raise ValueError('incomplete reference training replay')
    finally:
        if digest((dataset['normalizer'], type_manifest, frozen.manifest)) != protected:
            raise ValueError('reference preprocessing/provenance mutated')
    return results

"""Predeclared saved-market engineering replay; cannot promote or select models."""
from contextlib import contextmanager
from datetime import date
import hashlib
import json
from pathlib import Path
import time

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.provenance import repository_state
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.evaluation.canonical_task import evaluate_canonical_training
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.evaluation.training_fit import training_window
from rl_stocks.inference.canonical_bundle import FrozenCanonicalTaskPolicy, freeze_canonical_task
from rl_stocks.reporting.initialization import group_metrics

PROTOCOL_SHA256 = 'f5fb168a3e71ee071917f53697965e378296237bcb8dff1fce9719a1aca8d328'


def protocol(root):
    value = json.loads((Path(root) / 'configs/experiments/canonical_task_replay.json').read_text())
    if digest(value) != PROTOCOL_SHA256:
        raise ValueError('fixed reference market-replay protocol required')
    return value


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def inference_only():
    """Fail before any accidental native inference, optimization or backward."""
    from stable_baselines3 import SAC
    import torch

    def forbidden(*args, **kwargs):
        raise RuntimeError('reference diagnostic forbids native predict or local optimization')

    targets = [(SAC, name) for name in ('learn', 'train', 'predict')]
    targets += [(torch.Tensor, 'backward'), (torch.optim.Adam, 'step')]
    previous = [(owner, name, getattr(owner, name)) for owner, name in targets]
    try:
        for owner, name, _ in previous:
            setattr(owner, name, forbidden)
        yield
    finally:
        for owner, name, method in previous:
            setattr(owner, name, method)


def validate_window(view, spec):
    episodes = view['episodes']
    if (len(episodes) != spec['episodes']
            or digest(episodes) != spec['window_episodes_sha256']
            or sum(len(e['bars'])-1 for e in episodes.values()) != spec['predictions']):
        raise ValueError('exact fixed26 training windows required')
    for episode in episodes.values():
        bars = episode['bars']
        if (episode['split'] != 'train' or bars[0]['session'] != spec['anchor_session']
                or bars[1]['session'] != spec['first_execution_session']
                or bars[-1]['session'] != spec['last_session']
                or any(date.fromisoformat(b['session']) > date(2020, 12, 31) for b in bars)):
            raise ValueError('reference replay forbids development/test rows')


def run(root, output):
    root, output = Path(root), Path(output)
    spec = protocol(root)
    git = repository_state(root)
    if git['dirty']:
        raise ValueError('clean committed reference replay source required')
    source = (root / 'artifacts/colab' / spec['source_session'] / 'retrieved/artifacts'
              / spec['source_subdirectory'] / spec['source_bundle'])
    dataset_path = root / 'data/datasets' / spec['dataset']
    protected = {str(p): sha(p) for p in source.iterdir()}
    protected[str(dataset_path / 'dataset.json')] = sha(dataset_path / 'dataset.json')
    data = read_dataset(dataset_path)
    view = training_window(data, original_panel_only=True)
    validate_window(view, spec)
    if digest(data['normalizer']) != spec['normalizer_sha256']:
        raise ValueError('fixed reference preprocessing required')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        with inference_only():
            path = freeze_canonical_task(source, output / 'frozen')
            frozen = FrozenCanonicalTaskPolicy(path, feature_schema=data['schema'],
                feed=data['manifest']['provenance']['capabilities']['feed'])
            original = frozen.manifest['source_manifest']
            if (original['model_sha256'] != spec['source_model_sha256']
                    or frozen.task_role != spec['task_role']
                    or frozen.execution['evaluation_exposure_mode'] != spec['exposure_execution']):
                raise ValueError('fixed saved actor and execution required')
            evidence = original['metadata']['task_contract']['type_manifest']
            start_replay = time.monotonic()
            results = evaluate_canonical_training(frozen, data, evidence)
            replay_seconds = time.monotonic() - start_replay
        if (len(results) != spec['episodes']
                or sum(len(r['rows']) for r in results.values()) != spec['predictions']):
            raise ValueError('complete reference episode paths required')
        for key, result in results.items():
            if (result['split'] != 'train'
                    or [r['session'] for r in result['rows']] !=
                    [b['session'] for b in view['episodes'][key]['bars'][1:]]):
                raise ValueError('exact chronological training outcomes required')
        stocks = set(data['manifest']['evaluation_symbols']) - {'SPY', 'QQQ', 'IWM', 'DIA'}
        from rl_stocks.reporting.html import report
        report(results, output / 'replay.html', 'Reference actor — training-only engineering diagnostic')
        record = {'schema': 'canonical-market-replay-result-v1', 'protocol_sha256': PROTOCOL_SHA256,
                  'source': git, 'source_bundle': spec['source_bundle'], 'reference_bundle': path.name,
                  'source_model_sha256': spec['source_model_sha256'], 'predictions': spec['predictions'],
                  'episodes': len(results), 'replay_seconds': replay_seconds,
                  'elapsed_seconds': time.monotonic()-started, 'results_sha256': digest(results),
                  'report_sha256': sha(output / 'replay.html'), 'report_json_sha256': sha(output / 'replay.json'),
                  'metrics': group_metrics(results, stocks),
                  'short_side': declining_stock_summary(results, stocks),
                  'holding_path_sha256': digest({k: r['baselines']['buy_hold'] for k, r in results.items()}),
                  'protected_inputs': protected, 'protected_inputs_unchanged': True,
                  'training_performed': False, 'checkpoint_selected': False,
                  'development_evaluated': False, 'retired_test_access': False,
                  'cross_machine_simulator_accepted': False, 'performance_promotion': False,
                  'native_acceptance_changed': False, 'colab_allocated': False}
        if record['elapsed_seconds'] >= spec['deadline_seconds']:
            raise TimeoutError('reference diagnostic deadline exceeded')
        if any(sha(Path(name)) != expected for name, expected in protected.items()):
            raise ValueError('reference diagnostic mutated protected source inputs')
        (output / 'result.json').write_bytes(canonical(record))
        return record
    finally:
        if any(sha(Path(name)) != expected for name, expected in protected.items()):
            raise ValueError('reference diagnostic mutated protected source inputs')

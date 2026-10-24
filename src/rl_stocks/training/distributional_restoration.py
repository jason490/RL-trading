"""Read only task-owned retrieved checkpoints/replay; never fit a policy."""
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import torch

from rl_stocks.core.contracts import canonical
from rl_stocks.inference.bundle import FrozenPolicy, checksum
from rl_stocks.rl.registry.factory import CLASSES
from rl_stocks.training.distributional_critics import capture_global_rng, restore_global_rng
from rl_stocks.training.distributional_records import model_record, validate_record


def _same_restored(actual, recorded):
    # Raw log-alpha/weight/moment fingerprints remain exact. Only derived CPU
    # exp(log-alpha) may differ by float32 rounding between runtime hosts.
    left = json.loads(json.dumps({k: v for k, v in actual.items() if k != 'rng'}))
    right = json.loads(json.dumps({k: v for k, v in recorded.items() if k != 'rng'}))
    alpha_left = left['initialization'].pop('alpha')
    alpha_right = right['initialization'].pop('alpha')
    return (canonical(left) == canonical(right)
            and math.isclose(alpha_left, alpha_right, rel_tol=1e-6, abs_tol=0.0))


def verify_owned_run(run_path, bundle_path, arm, smoke=False, seed=11):
    """Caller must first verify the owned archive, source and expected job identity."""
    numpy_python_rng = capture_global_rng()
    with torch.random.fork_rng(devices=[]):
        try:
            return _verify(Path(run_path), Path(bundle_path), arm, smoke, seed)
        finally:
            restore_global_rng(numpy_python_rng)


def _verify(run_path, bundle_path, arm, smoke, seed):
    metadata_path = run_path / 'metadata.json'
    m = json.loads(metadata_path.read_text())
    paths = [metadata_path, run_path / 'latest.zip', run_path / 'best.zip', run_path / 'replay.pkl',
             bundle_path / 'manifest.json', bundle_path / 'model.zip']
    protected = {str(p): checksum(p) for p in paths}
    if (m['status'] != 'complete' or m['critic_arm'] != arm or m['smoke'] is not smoke or m['seed'] != seed):
        raise ValueError('owned completed critic run identity mismatch')
    algorithm = m['algorithm']
    models = {}
    for name, filename, field in [('latest', 'latest.zip', 'critic_final'), ('selected', 'best.zip', 'critic_selected')]:
        candidate = CLASSES[algorithm].load(run_path / filename, device='cpu')
        candidate.policy.set_training_mode(False)
        record = model_record(candidate)
        validate_record(record, arm, smoke, selected_steps=m['selected_step'] if name == 'selected' else None, seed=seed)
        if not _same_restored(record, m[field]):
            raise ValueError('actual saved critic weights/optimizer/provenance did not restore')
        models[name] = candidate
    latest = models['latest']
    latest.load_replay_buffer(run_path / 'replay.pkl')  # Only the task-owned validated transport.
    buffer = latest.replay_buffer
    steps = m['actual_steps']
    if (type(buffer).__name__ != 'ReplayBuffer' or buffer.n_envs != 1 or buffer.pos != steps or buffer.full
            or buffer.optimize_memory_usage or not buffer.handle_timeout_termination
            or buffer.observations.shape[1:] != (1, 21) or buffer.actions.shape[1:] != (1, 4)):
        raise ValueError('complete fixed one-step critic replay required')
    for values in (buffer.observations, buffer.next_observations, buffer.actions, buffer.rewards):
        if not np.isfinite(values[:steps]).all():
            raise ValueError('finite actual replay rows required')
    done, timeout = buffer.dones[:steps], buffer.timeouts[:steps]
    if not np.isin(done, [0, 1]).all() or not np.isin(timeout, [0, 1]).all() or (timeout > done).any():
        raise ValueError('actual replay termination/timeout mask mismatch')
    if not _same_restored(model_record(latest), m['critic_final']):
        raise ValueError('replay loading changed learned critic state')
    manifest = json.loads((bundle_path / 'manifest.json').read_text())
    feed = manifest['data_manifest']['provenance']['capabilities']['feed']
    frozen = FrozenPolicy(bundle_path, manifest['feature_schema'], feed, 'cpu')
    before = canonical(asdict(frozen.normalizer))
    if not _same_restored(model_record(frozen._model), m['critic_selected']):
        raise ValueError('complete frozen critic/optimizer state mismatch')
    indices = np.linspace(0, steps - 1, 64, dtype=int)
    for observation in buffer.observations[indices, 0]:
        expected = models['selected'].predict(observation, deterministic=True)[0]
        np.testing.assert_array_equal(frozen.predict(observation), expected)
    if canonical(asdict(frozen.normalizer)) != before:
        raise ValueError('frozen critic normalization changed')
    if not _same_restored(model_record(frozen._model), m['critic_selected']):
        raise ValueError('frozen prediction changed weights or optimizer state')
    if any(checksum(p) != value for p, value in protected.items()):
        raise ValueError('owned critic source files changed during verification')
    return dict(actual_saved_models_restored=True, latest_selected_and_frozen_optimizers_exact=True,
                replay_verified=True, replay_rows=steps, exact_frozen_predictions=len(indices),
                normalization_unchanged=True, protected_source_sha256=protected,
                optimizer_steps=0, source_files_unchanged=True)

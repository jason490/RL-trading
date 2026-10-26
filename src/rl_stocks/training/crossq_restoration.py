"""Read only task-owned retrieved checkpoints/replay; never fit a policy."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from rl_stocks.core.contracts import canonical
from rl_stocks.inference.bundle import FrozenPolicy, checksum
from rl_stocks.rl.registry.factory import CLASSES
from rl_stocks.training.distributional_critics import capture_global_rng, restore_global_rng
from rl_stocks.training.crossq_state import model_record, validate_record
from rl_stocks.training.crossq_replay import replay_record
from rl_stocks.training.recurrent_restore import same_restored as _same_restored


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
             run_path / 'crossq-replay-state.json', bundle_path / 'manifest.json', bundle_path / 'model.zip',
             bundle_path / 'checksums.json']
    paths += [p for p in bundle_path.iterdir() if p.is_file() and p not in paths]
    protected = {str(p): checksum(p) for p in paths}
    if (m['status'] != 'complete' or m['crossq_arm'] != arm or m['smoke'] is not smoke or m['seed'] != seed):
        raise ValueError('owned completed CrossQ run identity mismatch')
    algorithm = m['algorithm']
    if algorithm != arm or arm not in ('sac', 'crossq'):
        raise ValueError('fixed SAC/CrossQ native model identity required')
    models = {}
    for name, filename, field in [('latest', 'latest.zip', 'crossq_final'), ('selected', 'best.zip', 'crossq_selected')]:
        candidate = CLASSES[algorithm].load(run_path / filename, device='cpu')
        candidate.policy.set_training_mode(False)
        record = model_record(candidate)
        validate_record(record, arm, smoke, phase=m['crossq_selected_phase'] if name == 'selected' else 'post_update')
        wanted_steps = m['selected_step'] if name == 'selected' else m['actual_steps']
        if record['num_timesteps'] != wanted_steps or record['seed'] != seed:
            raise ValueError('saved CrossQ checkpoint transition/seed mismatch')
        if not _same_restored(record, m[field]):
            raise ValueError('actual saved CrossQ weights/optimizer/provenance did not restore')
        models[name] = candidate
    latest = models['latest']
    latest.load_replay_buffer(run_path / 'replay.pkl')  # Only the task-owned validated transport.
    buffer = latest.replay_buffer
    steps = m['actual_steps']
    replay = replay_record(buffer, steps)
    if canonical(replay) != canonical(m['crossq_replay']):
        raise ValueError('actual saved replay rows/timeouts differ from recorded evidence')
    if canonical(replay) != canonical(json.loads((run_path / 'crossq-replay-state.json').read_text())):
        raise ValueError('actual saved replay sidecar mismatch')
    if not _same_restored(model_record(latest), m['crossq_final']):
        raise ValueError('replay loading changed learned CrossQ state')
    manifest = json.loads((bundle_path / 'manifest.json').read_text())
    feed = manifest['data_manifest']['provenance']['capabilities']['feed']
    frozen = FrozenPolicy(bundle_path, manifest['feature_schema'], feed, 'cpu')
    before = canonical(asdict(frozen.normalizer))
    with np.testing.assert_raises(ValueError):
        FrozenPolicy(bundle_path, 'invalid-schema', feed, 'cpu')
    with np.testing.assert_raises(ValueError):
        FrozenPolicy(bundle_path, manifest['feature_schema'], 'invalid-feed', 'cpu')
    if not _same_restored(model_record(frozen._model), m['crossq_selected']):
        raise ValueError('complete frozen CrossQ/optimizer state mismatch')
    indices = np.linspace(0, steps - 1, 64, dtype=int)
    for observation in buffer.observations[indices, 0]:
        original = observation.copy()
        expected = models['selected'].predict(observation, deterministic=True)[0]
        np.testing.assert_array_equal(frozen.predict(observation), expected)
        np.testing.assert_array_equal(frozen.predict(observation), expected)
        np.testing.assert_array_equal(observation, original)
    if canonical(asdict(frozen.normalizer)) != before:
        raise ValueError('frozen CrossQ normalization changed')
    if not _same_restored(model_record(frozen._model), m['crossq_selected']):
        raise ValueError('frozen prediction changed weights or optimizer state')
    if any(checksum(p) != value for p, value in protected.items()):
        raise ValueError('owned CrossQ source files changed during verification')
    return dict(actual_saved_models_restored=True, latest_selected_and_frozen_optimizers_exact=True,
                replay_verified=True, final_replay_restored=True, replay_rows=steps,
                exact_frozen_predictions=2 * len(indices), frozen_predictions=2 * len(indices),
                schema_and_feed_rejected=True,
                deterministic_repeat_and_input_immutability=True,
                normalization_unchanged=True, protected_source_sha256=protected,
                optimizer_steps=0, source_files_unchanged=True)

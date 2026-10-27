"""Physical native save/load and filled replay fixtures; no optimization."""
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from test_crossq_state import populated, no_optimization  # noqa: F401
from rl_stocks.core.contracts import canonical
from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.environment.observations.state import observation_spec
from rl_stocks.features.registry.assemble import NAMES
from rl_stocks.inference.bundle import freeze
from rl_stocks.risk.exposure import execution_contract
from rl_stocks.training.crossq_replay import replay_record
from rl_stocks.training.crossq_restoration import verify_owned_run
from rl_stocks.training.crossq_state import model_record
from rl_stocks.training.distributional_critics import rng_record


def saved_fixture(tmp_path, arm, phase='post_update'):
    run = tmp_path / 'run'
    run.mkdir()
    latest = populated(arm)
    selected = latest if phase == 'post_update' else populated(arm, 2048, 'on_step')
    buffer = latest.replay_buffer
    for field in ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts'):
        getattr(buffer, field)[:4096] = 0.
    buffer.observations[:4096, 0, 0] = np.linspace(-1, 1, 4096)
    buffer.dones[10] = buffer.timeouts[10] = 1.
    buffer.dones[20] = 1.
    buffer.pos = 4096
    latest.save(run / 'latest.zip')
    selected.save(run / 'best.zip')
    latest.save_replay_buffer(run / 'replay.pkl')
    record = model_record(selected)
    metadata = dict(status='complete', algorithm=arm, crossq_arm=arm, seed=11, smoke=True,
        actual_steps=4096, selected_step=selected.num_timesteps, crossq_selected_phase=phase,
        crossq_final=model_record(latest), crossq_selected=record, crossq_replay=replay_record(buffer, 4096),
        observation=observation_spec('execution_full'), policy_architecture={'version': 1, 'variant': 'native'},
        action_execution=execution_contract(), action_distribution={'version': 1, 'kind': 'squashed_gaussian', 'components': 4})
    if arm == 'crossq':
        metadata.update(crossq_geometry=record['geometry'], crossq_normalization=record['normalization'])
    data = dict(schema='daily-indicators-v1', names=list(NAMES),
        normalizer=asdict(Normalizer(tuple(NAMES), (0.,) * 7, (1.,) * 7)),
        manifest={'included_splits': ['train', 'validation'], 'provenance': {'capabilities': {'feed': 'fixture'}}})
    bundle = freeze(run / 'best.zip', data, metadata, tmp_path / 'frozen')
    metadata['bundle_id'] = bundle.name
    (run / 'metadata.json').write_text(json.dumps(metadata))
    (run / 'crossq-replay-state.json').write_text(json.dumps(metadata['crossq_replay']))
    return run, bundle, latest


@pytest.mark.parametrize('arm', ['sac', 'crossq'])
@pytest.mark.parametrize('phase', ['on_step', 'post_update'])
def test_native_models_adam_replay_frozen_predictions_and_rng_restore(tmp_path, arm, phase):
    run, bundle, _ = saved_fixture(tmp_path, arm, phase)
    before = canonical(rng_record())
    result = verify_owned_run(run, bundle, arm, smoke=True)
    assert canonical(rng_record()) == before
    assert result['actual_saved_models_restored'] and result['final_replay_restored']
    assert result['exact_frozen_predictions'] == 128
    assert result['optimizer_steps'] == 0
    assert result['source_files_unchanged'] and result['schema_and_feed_rejected']


@pytest.mark.parametrize('change', ['filled_row', 'timeout', 'optimizer', 'normalization', 'selected_phase'])
def test_physical_changes_cannot_hide_behind_nominal_metadata(tmp_path, change):
    run, bundle, latest = saved_fixture(tmp_path, 'crossq')
    if change == 'filled_row':
        latest.replay_buffer.observations[15, 0, 0] += 1.
        latest.save_replay_buffer(run / 'replay.pkl')
    elif change == 'timeout':
        latest.replay_buffer.timeouts[10] = 0.
        latest.save_replay_buffer(run / 'replay.pkl')
    elif change == 'optimizer':
        latest.actor.optimizer.state[next(latest.actor.parameters())]['exp_avg'].add_(1.)
        latest.save(run / 'latest.zip')
    elif change == 'normalization':
        with torch.no_grad():
            latest.actor.latent_pi[0].ra_mean.add_(1.)
        latest.save(run / 'latest.zip')
    else:
        path = run / 'metadata.json'
        metadata = json.loads(path.read_text())
        metadata['crossq_selected_phase'] = 'on_step'
        path.write_text(json.dumps(metadata))
    before = canonical(rng_record())
    with pytest.raises(ValueError):
        verify_owned_run(run, bundle, 'crossq', smoke=True)
    assert canonical(rng_record()) == before


def test_only_derived_exp_may_differ_by_one_float32_ulp():
    from rl_stocks.training.crossq_restoration import _same_restored

    value = float(np.float32(0.66))
    adjacent = float(np.nextafter(np.float32(value), np.float32(np.inf)))
    further = float(np.nextafter(np.float32(adjacent), np.float32(np.inf)))
    record = dict(alpha=value, exact_tensor_sha256='same', rng={})
    assert _same_restored(dict(record, alpha=adjacent), record)
    assert not _same_restored(dict(record, alpha=further), record)
    assert not _same_restored(dict(record, alpha=adjacent, exact_tensor_sha256='changed'), record)
    assert not _same_restored(dict(record, alpha=float('nan')), record)

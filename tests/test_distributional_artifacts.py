"""Owned serialization/archive fixtures; no fitting and no performance evidence."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tarfile

import pytest
import torch

from test_distributional_initialization import model
from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import Normalizer
from rl_stocks.environment.observations.state import observation_spec
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.features.registry.assemble import NAMES
from rl_stocks.inference.bundle import freeze
from rl_stocks.risk.exposure import execution_contract
from rl_stocks.rl.registry.factory import resolved_parameters
from rl_stocks.runtime.colab import distributional_artifacts as contracts
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.runtime.colab.jobs import critic_plan
from rl_stocks.training.checkpoint_state import write_checkpoint_state
from rl_stocks.training.distributional_assets import initialization_entry
from rl_stocks.training.distributional_critics import apply_initialization, overrides
from rl_stocks.training.distributional_records import model_record
from rl_stocks.training.distributional_restoration import verify_owned_run


def archive(root, monkeypatch, arm='tqc_truncated', mutate=None, replay=False):
    plan = critic_plan(arm, smoke=True)
    job = plan['jobs'][0]
    run = root / 'artifacts/runs/fixture'
    run.mkdir(parents=True)
    m = model(arm)
    proof = apply_initialization(m, arm, smoke=True)
    initial = model_record(m)
    shutil.copyfile(Path(__file__).parents[1] / initialization_entry(arm, smoke=True)[0]['file'],
                    run / 'critic-initialization.npz')
    # Complete nonempty serialization fixture, never produced by optimizer.step.
    for opt in (m.actor.optimizer, m.critic.optimizer, m.ent_coef_optimizer):
        for group in opt.param_groups:
            for parameter in group['params']:
                opt.state[parameter] = dict(step=torch.tensor(960.), exp_avg=torch.ones_like(parameter),
                                           exp_avg_sq=torch.full_like(parameter, 0.5))
    with torch.no_grad():
        next(m.actor.parameters()).add_(0.01)
    m.num_timesteps, m._n_updates = 1024, 960
    final = model_record(m)
    m.save(run / 'best.zip')
    m.save(run / 'latest.zip')
    if replay:
        m.replay_buffer.pos = 1024
        m.save_replay_buffer(run / 'replay.pkl')
    else:
        (run / 'replay.pkl').write_bytes(b'archive-only fixture; not deserialized')
    counts = dict(mode='unrestricted', transitions=1024, requested_nonzero=1024,
                  processed_nonunit=1024, sizing_changed=0, protective_changed=0)
    curve = [dict(steps=512, phase='on_step', score=0.01), dict(steps=1024, phase='post_update', score=0.01)]
    metadata = dict(algorithm=job['algorithm'], device='cpu', smoke=True, seed=11, dataset=plan['dataset_contract']['identity'],
        status='complete', actual_steps=1024, requested_steps=1024, selected_step=1024, duration_seconds=10.,
        simulation=asdict(SimulationConfig()), observation=observation_spec('execution_full'), initialization={'arm': 'standard', 'applied': False},
        start_sampling={'mode': 'legacy_uniform'}, sampling_diagnostics=None,
        policy_architecture={'version': 1, 'variant': 'native'},
        resolved_parameters=resolved_parameters(job['algorithm'], True, overrides(arm, True)),
        reward_config={'kind': 'net_log_equity', 'scale': 1.0}, milestones=[1024], evaluation_every=512,
        compact_checkpoints=True, validation_checkpoints=curve, action_execution=execution_contract(),
        action_distribution={'version': 1, 'kind': 'squashed_gaussian', 'components': 4},
        study_protocol_sha256=plan['protocol_sha256'], critic_arm=arm,
        critic_shared_initialization=proof, critic_initialization=initial, critic_final=final,
        critic_selected=deepcopy(final), training_execution=counts)
    if job['algorithm'] == 'tqc':
        metadata['critic_geometry'] = final['critic_geometry']
    normalizer = json.loads(json.dumps(asdict(Normalizer(tuple(NAMES), (0.,) * 7, (1.,) * 7))))
    monkeypatch.setattr(contracts, 'NORMALIZER_SHA256', digest(normalizer))
    data = dict(schema='daily-indicators-v1', names=list(NAMES), normalizer=normalizer,
                manifest={'included_splits': ['train', 'validation'], 'provenance': {'capabilities': {'feed': 'fixture'}}})
    if mutate:
        mutate(metadata, data)
    bundle = freeze(run / 'best.zip', data, metadata, root / 'artifacts/frozen')
    metadata['bundle_id'] = bundle.name
    (run / 'metadata.json').write_text(json.dumps(metadata))
    (run / 'training-execution.json').write_text(json.dumps(counts))
    (run / 'learning-curve.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in curve))
    (run / 'learning-curve.csv').write_text('steps,return_pct\n512,1.0\n1024,1.0\n')
    (run / 'learning-curve.html').write_text('<html>Fixture only</html>')
    reports = {symbol: {'symbol': symbol, 'split': 'validation', 'metrics': {'net_return': 0.01}}
               for symbol in contracts.DEVELOPMENT_SYMBOLS}
    for filename in ['selected-validation.json', 'validation.json']:
        (run / filename).write_text(json.dumps(reports))
    write_checkpoint_state(run, reports, 'latest.zip', 1024, 'post_update', 1024, 'post_update', True)
    completion = dict(plan=plan, dataset=metadata['dataset'], python=[3, 13, 16], gpu=None,
                      completed=[{'bundle_id': bundle.name}])
    (root / 'artifacts/completion.json').write_text(json.dumps(completion))
    packed = root / 'fixture.tar'
    with tarfile.open(packed, 'w') as output:
        output.add(root / 'artifacts', arcname='artifacts')
    return packed, plan, metadata['dataset'], run, bundle


@pytest.mark.parametrize('arm', ['sac_native', 'tqc_untruncated', 'tqc_truncated'])
def test_valid_critic_archive_and_actual_serializer_replay_frozen_parity(tmp_path, monkeypatch, arm):
    packed, plan, dataset, run, bundle = archive(tmp_path, monkeypatch, arm, replay=True)
    assert len(validate_archive(packed, plan, dataset)) == 1
    proof = verify_owned_run(run, bundle, arm, smoke=True)
    assert proof['actual_saved_models_restored'] and proof['replay_rows'] == 1024
    assert proof['exact_frozen_predictions'] == 64 and proof['optimizer_steps'] == 0


@pytest.mark.parametrize('change', [
    lambda m, d: m.update(critic_arm='sac_native'),
    lambda m, d: m['critic_final'].update(updates=959),
    lambda m, d: m['critic_selected']['adam_state']['entropy'].update(state_count=0),
    lambda m, d: m['critic_shared_initialization'].update(rng_sha256='0' * 64),
    lambda m, d: m['critic_geometry'].update(n_quantiles=24),
    lambda m, d: m['simulation'].update(fee_bps=0.),
    lambda m, d: m.update(action_distribution={'version': 1, 'kind': 'diagonal_gaussian', 'components': 4}),
    lambda m, d: d['normalizer']['means'].__setitem__(0, 123),
])
def test_checksums_do_not_excuse_semantic_critic_metadata_tampering(tmp_path, monkeypatch, change):
    packed, plan, dataset, _, _ = archive(tmp_path, monkeypatch, mutate=change)
    with pytest.raises(RuntimeError):
        validate_archive(packed, plan, dataset)


@pytest.mark.parametrize('missing', ['critic-initialization.npz', 'replay.pkl', 'latest.zip', 'checkpoint-state.json'])
def test_mandatory_complete_critic_evidence_cannot_be_omitted(tmp_path, monkeypatch, missing):
    _, plan, dataset, run, _ = archive(tmp_path, monkeypatch)
    (run / missing).unlink()
    packed = tmp_path / 'missing.tar'
    with tarfile.open(packed, 'w') as output:
        output.add(tmp_path / 'artifacts', arcname='artifacts')
    with pytest.raises(RuntimeError):
        validate_archive(packed, plan, dataset)

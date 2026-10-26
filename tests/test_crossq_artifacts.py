"""Archive semantic fixtures with manual Adam counters; no fitting/deserialization."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import pytest

from test_crossq_state import model, populated, no_optimization  # noqa: F401
from rl_stocks.core.contracts import digest
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.features.registry.assemble import NAMES
from rl_stocks.risk.exposure import execution_contract
from rl_stocks.runtime.colab import crossq_artifacts as contracts
from rl_stocks.runtime.colab.jobs import crossq_plan
from rl_stocks.training.crossq_assets import initialization_entry, portable_path
from rl_stocks.training.crossq_initialization import apply_initialization
from rl_stocks.training.crossq_state import model_record
from rl_stocks.training.crossq_forward_audit import SOURCE_SHA256
from rl_stocks.training.crossq_spec import expected_updates


def fixture(monkeypatch, arm):
    plan = crossq_plan(arm, smoke=True)
    initial_model = model(arm)
    proof = apply_initialization(initial_model, arm, smoke=True)
    initial = model_record(initial_model)
    final = model_record(populated(arm))
    counts = dict(mode='unrestricted', transitions=4096, requested_nonzero=4096,
                  processed_nonunit=4096, sizing_changed=0, protective_changed=0)
    run = dict(crossq_arm=arm, crossq_binding_sha256=plan['binding_sha256'],
        study_protocol_sha256=plan['protocol_sha256'], duration_seconds=10.,
        start_sampling={'mode': 'legacy_uniform'}, sampling_diagnostics=None,
        action_execution=execution_contract(), action_distribution={'version': 1, 'kind': 'squashed_gaussian', 'components': 4},
        simulation=asdict(SimulationConfig()), crossq_shared_initialization=proof,
        crossq_initialization=initial, crossq_final=final, crossq_selected=deepcopy(final),
        crossq_selected_phase='post_update', selected_step=4096,
        training_execution=counts, crossq_replay={'filled_transitions': 4096}, validation_checkpoints=[])
    files = {'run/training-execution.json': counts, 'run/checkpoint-state.json': {'selected': {'phase': 'post_update'}},
             'run/crossq-replay-state.json': run['crossq_replay']}
    if arm == 'crossq':
        updates = expected_updates(arm, 4096, 'post_update', True)
        audit = dict(version=1, source_sha256=SOURCE_SHA256, observed=True, warmup_steps=1000, layers={})
        for name in final['normalization']:
            branch = 'actor' if name.startswith('actor.') else 'critic'
            n = updates[branch + '_normalization']
            audit['layers'][name] = dict(training_calls=n, evaluation_calls=0, counter=n,
                                         renorm_condition_calls=updates[branch + '_renorm_forward_calls'])
        run.update(crossq_geometry=final['geometry'], crossq_normalization=final['normalization'], crossq_forward_audit=audit)
        files['run/crossq-normalization-audit.json'] = deepcopy(audit)
        run['validation_checkpoints'] = [dict(steps=4096, phase='post_update', crossq_forward_audit=deepcopy(audit))]
    reports = {s: {'symbol': s, 'split': 'validation'} for s in contracts.DEVELOPMENT_SYMBOLS}
    files.update({'run/selected-validation.json': reports, 'run/validation.json': reports})
    normalizer = {'serialization_fixture': True}
    monkeypatch.setattr(contracts, 'NORMALIZER_SHA256', digest(normalizer))
    run = json.loads(json.dumps(run))
    files['frozen/manifest.json'] = dict(metadata=deepcopy(run), names=list(NAMES),
        feature_schema='daily-indicators-v1', normalizer=normalizer,
        data_manifest={'included_splits': ['train', 'validation']}, model_sha256=hashlib.sha256(b'model').hexdigest())
    files = {k: json.dumps(v).encode() for k, v in files.items()}
    entry, _ = initialization_entry(arm, smoke=True)
    files['run/crossq-initialization.npz'] = portable_path(entry['file']).read_bytes()
    files.update({'run/best.zip': b'model', 'frozen/model.zip': b'model', 'run/latest.zip': b'final', 'run/replay.pkl': b'not deserialized here'})
    return run, plan, files


@pytest.mark.parametrize('arm', ['sac', 'crossq'])
def test_actual_state_semantics_with_manual_serialization_fixture(monkeypatch, arm):
    run, plan, files = fixture(monkeypatch, arm)
    contracts.validate_crossq_artifacts(files.__getitem__, 'run/', 'frozen/', run, plan)


@pytest.mark.parametrize('change', ['phase', 'selected_norm', 'observed', 'replay', 'initializer', 'model', 'selected_audit'])
def test_archive_binding_rejects_mutation(monkeypatch, change):
    run, plan, files = fixture(monkeypatch, 'crossq')
    if change == 'phase':
        run['crossq_selected_phase'] = 'on_step'
    elif change == 'selected_norm':
        run['crossq_normalization'] = {}
    elif change == 'observed':
        run['crossq_forward_audit']['observed'] = False
    elif change == 'replay':
        files['run/crossq-replay-state.json'] = b'{}'
    elif change == 'selected_audit':
        run['validation_checkpoints'] = []
    elif change == 'initializer':
        files['run/crossq-initialization.npz'] = b'invalid'
    else:
        files['frozen/model.zip'] = b'changed'
    with pytest.raises((RuntimeError, ValueError)):
        contracts.validate_crossq_artifacts(files.__getitem__, 'run/', 'frozen/', run, plan)

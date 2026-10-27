"""Incomplete/recovered fleet states never become learned smoke evidence."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('accept_crossq', Path(__file__).parents[1] / 'scripts/accept_crossq_smokes.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('change', ['running', 'cleanup', 'missing_arm', 'extra_session', 'unfinished'])
def test_reject_incomplete_or_unowned_smoke_state(tmp_path, change):
    tasks = ['crossq-sac-11-smoke', 'crossq-crossq-11-smoke']
    state = dict(status='complete', cleanup_verified=True, completed={t: {} for t in tasks},
                 sessions={str(i): dict(task=t, phase='done') for i, t in enumerate(tasks)})
    if change == 'running':
        state['status'] = 'running'
    elif change == 'cleanup':
        state['cleanup_verified'] = False
    elif change == 'missing_arm':
        state['completed'].pop(tasks[0])
    elif change == 'extra_session':
        state['sessions']['other'] = dict(task='unrelated', phase='done')
    else:
        state['sessions']['0']['phase'] = 'stopping'
    state_path, source_path = tmp_path / 'state.json', tmp_path / 'source.json'
    state_path.write_text(json.dumps(state))
    source_path.write_text('{}')
    with pytest.raises(ValueError):
        module.accept(state_path, source_path, tmp_path)


@pytest.mark.parametrize('mutate', [False, True])
def test_completion_plan_json_normalization_preserves_exact_semantics(tmp_path, monkeypatch, mutate):
    from rl_stocks.runtime.colab.jobs import crossq_plan
    from rl_stocks.training.crossq_spec import PROTOCOL_SHA256, protocol

    tasks = ['crossq-sac-11-smoke', 'crossq-crossq-11-smoke']
    artifact = tmp_path / 'artifacts'
    run = artifact / 'runs/fixture'
    run.mkdir(parents=True)
    state = dict(status='complete', cleanup_verified=True,
        sessions={str(i): dict(task=t, phase='done') for i, t in enumerate(tasks)},
        completed={t: dict(session=str(i), root=str(artifact)) for i, t in enumerate(tasks)})
    source = dict(source_revision='fixture', source_hash='fixture')
    metadata = dict(git=dict(revision='fixture', dirty=''), source_hash='fixture',
        study_protocol_sha256=PROTOCOL_SHA256, dataset=protocol()['dataset'],
        smoke=True, seed=11, actual_steps=4096)
    (run / 'metadata.json').write_text(json.dumps(metadata))
    plan = crossq_plan('sac', smoke=True)
    if mutate:
        plan['jobs'][0]['steps'] = 4095
    (artifact / 'completion.json').write_text(json.dumps(dict(plan=plan)))
    state_path, source_path = tmp_path / 'state.json', tmp_path / 'source.json'
    state_path.write_text(json.dumps(state))
    source_path.write_text(json.dumps(source))
    monkeypatch.setattr(module, 'read_dataset', lambda path: {})
    monkeypatch.setattr(module, 'validate_payload', lambda data: data)

    def reached_archive(*args):
        raise RuntimeError('exact completion reached archive validation')

    monkeypatch.setattr(module, 'validate_archive', reached_archive)
    if mutate:
        with pytest.raises(ValueError, match='completion plan mismatch'):
            module.accept(state_path, source_path, tmp_path)
    else:
        with pytest.raises(RuntimeError, match='reached archive validation'):
            module.accept(state_path, source_path, tmp_path)

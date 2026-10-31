import importlib
import json
from pathlib import Path

import pytest


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    return importlib.import_module('resume_task_stock_sources')


def test_resume_rejects_changed_plan_before_identity_or_network(tmp_path, monkeypatch):
    worker = module(monkeypatch)
    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    monkeypatch.setattr(worker, 'prepare', lambda: {'different': True})
    def forbidden(*args, **kwargs):
        pytest.fail('identity/network used before reviewed-plan verification')
    monkeypatch.setattr(worker, 'identity', forbidden)
    monkeypatch.setattr(worker, 'acquire_filing_html', forbidden)
    with pytest.raises(ValueError, match='differs'):
        worker.execute(plan, tmp_path / 'result.json')
    assert not (tmp_path / 'result.json').exists()


def test_resume_failure_retains_old_stop_and_does_not_retry(tmp_path, monkeypatch):
    worker = module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    runtime = tmp_path / '.runtime'
    runtime.mkdir()
    old_stop = runtime / 'task-stock-master-collection-stop-v1.json'
    old_stop.write_text('original failure must remain unchanged')
    plan = {'cases': [{'symbol': 'A', 'status': 'not_attempted',
                       'master_record': {'filing_directory_cik': '1',
                                         'accession': '0000000001-10-000001'}}],
            'protected_sha256': {str(old_stop): worker.checksum(old_stop)}}
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(worker, 'prepare', lambda: plan)
    monkeypatch.setattr(worker, 'identity', lambda: 'synthetic request identity')
    calls = []
    def fail(*args, **kwargs):
        calls.append(True)
        raise ValueError('synthetic source rejection')
    monkeypatch.setattr(worker, 'acquire_filing_html', fail)
    output = tmp_path / 'result.json'
    with pytest.raises(RuntimeError, match='stopped'):
        worker.execute(plan_path, output)
    result = json.loads(output.read_text())
    assert len(calls) == 1 and result['status'] == 'stopped'
    assert result['protected_sources_unchanged']
    assert old_stop.read_text() == 'original failure must remain unchanged'
    assert (runtime / 'task-stock-reviewed-resume-v2-stop.json').is_file()
    assert not list(tmp_path.glob('*.tmp'))
    with pytest.raises(ValueError, match='already claimed'):
        worker.execute(plan_path, tmp_path / 'different.json')
    assert len(calls) == 1

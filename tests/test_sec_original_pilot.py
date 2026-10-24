import importlib.util
import json
from pathlib import Path

import pytest


def module(monkeypatch):
    directory = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('original_pilot', directory / 'collect_original_flow_pilot.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_stratified_selection_retains_wholly_absent_issuers_and_no_replacement(monkeypatch):
    m = module(monkeypatch)
    rows = [dict(cik='1', accession='later', archive_filed='20110201', form='10-K'),
            dict(cik='1', accession='early', archive_filed='20100201', form='10-K'),
            dict(cik='1', accession='amendment', archive_filed='20090201', form='10-K/A')]
    selected, absent = m.select(dict(records=rows), [[2009, 2011], [2021, 2023]], ['1', '2'])
    assert [r['accession'] for r in selected] == ['early']
    assert absent == [dict(cik='1', stratum='2021-2023'), dict(cik='2', stratum='2009-2011'),
                      dict(cik='2', stratum='2021-2023')]


@pytest.mark.parametrize('error', [ValueError('access rejected'), KeyboardInterrupt()])
def test_failure_terminal_state_and_no_automatic_restart(tmp_path, monkeypatch, error):
    m = module(monkeypatch)
    spec = json.loads(m.PROTOCOL.read_text())
    monkeypatch.chdir(tmp_path)
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps(spec))
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text('{}')
    monkeypatch.setattr(m, 'PROTOCOL', protocol)
    monkeypatch.setattr(m, 'MANIFEST', manifest_path)
    monkeypatch.setattr(m, 'STATE', tmp_path / 'state.json')
    monkeypatch.setattr(m, 'prepare', lambda: dict(selected=[dict(accession='one', cik='1', stratum='early')]))
    monkeypatch.setattr(m, 'identity', lambda: 'Research private@example.org')
    import subprocess
    monkeypatch.setattr(subprocess, 'check_output', lambda *args, **kwargs: 'fixed-revision')
    calls = []
    def fail(*args):
        calls.append(True)
        raise error
    monkeypatch.setattr(m, 'acquire_original', fail)
    with pytest.raises(type(error)):
        m.collect()
    state = json.loads(m.STATE.read_text())
    assert state['status'] == ('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
    assert not state['completed'] and state['current_accession'] == 'one'
    assert 'private@example.org' not in m.STATE.read_text()
    with pytest.raises(ValueError, match='automatic restart disabled'):
        m.collect()
    assert len(calls) == 1


def test_reviewed_resume_preserves_original_journal_and_completed_prefix(tmp_path, monkeypatch):
    m = module(monkeypatch)
    import hashlib
    source = tmp_path / 'source.xml'
    source.write_bytes(b'original')
    facts = tmp_path / 'facts.json'
    facts.write_text('{}')
    completed = [dict(cik='1', accession='first', stratum='early', status='inspected',
                      source_path=str(source), source_sha256=hashlib.sha256(b'original').hexdigest(),
                      bytes=8, fact_payload_path=str(facts), fact_payload_sha256=m.digest(facts))]
    manifest = dict(selected=[dict(cik='1', accession='first', stratum='early'),
                              dict(cik='1', accession='oversized', stratum='late')])
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text('{}')
    prior = dict(status='failed', error_type='ValueError', current_accession='oversized',
                 manifest_sha256=m.digest(manifest_path), source_git_revision='old',
                 completed=completed, total_source_bytes=8, started_utc_epoch=100)
    prior_path = tmp_path / 'prior.json'
    prior_path.write_text(json.dumps(prior))
    old = prior_path.read_bytes()
    addendum = tmp_path / 'resume.json'
    addendum.write_text(json.dumps(dict(prior_state_sha256=m.digest(prior_path),
                        prior_stop_accession='oversized', prior_rejected_response_bytes=20000001)))
    monkeypatch.setattr(m, 'STATE', prior_path)
    monkeypatch.setattr(m, 'MANIFEST', manifest_path)
    monkeypatch.setattr(m, 'RESUME_PROTOCOL', addendum)
    state = m.reviewed_start(manifest)
    assert prior_path.read_bytes() == old
    assert state['completed'][0] == completed[0]
    assert state['completed'][1]['status'] == 'source_exceeds_limit'
    assert state['completed'][1]['arithmetic_ready'] is False
    assert state['rejected_response_bytes'] == 20000001
    assert state['total_source_bytes'] == 8 and state['started_utc_epoch'] == 100
    source.write_bytes(b'changed!')
    with pytest.raises(ValueError, match='corruption'):
        m.reviewed_start(manifest)

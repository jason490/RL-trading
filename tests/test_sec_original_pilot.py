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

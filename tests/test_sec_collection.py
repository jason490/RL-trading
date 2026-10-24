import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    spec=importlib.util.spec_from_file_location(
        'sec_collection',Path(__file__).parents[1]/'scripts/collect_sec_archives.py')
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_all_quarters_journaled_without_identity_or_training(tmp_path,monkeypatch):
    m=module()
    def fake(root,quarter,user_agent):
        raw=quarter.encode()
        p=root/hashlib.sha256(raw).hexdigest()/'archive.zip'
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(raw)
        return p
    monkeypatch.setattr(m,'download_archive',fake)
    journal=tmp_path/'state.json'
    state=m.collect(tmp_path/'cache',journal,'private-contact@example.org')
    assert state['status']=='downloaded_only'
    assert len(state['completed'])==59
    assert state['completed'][0]['quarter']=='2009q2'
    assert state['completed'][-1]['quarter']=='2023q4'
    assert 'private-contact' not in journal.read_text()
    assert not state['training_performed'] and not state['colab_allocated']
    with pytest.raises(ValueError,match='automatic restart disabled'):
        m.collect(tmp_path/'cache',journal,'private-contact@example.org')


def test_failed_download_leaves_terminal_journal_without_retry(tmp_path,monkeypatch):
    m=module()
    calls=[]
    def fail(root,quarter,user_agent):
        calls.append(quarter)
        raise ValueError('access rejected')
    monkeypatch.setattr(m,'download_archive',fail)
    journal=tmp_path/'state.json'
    with pytest.raises(ValueError,match='access rejected'):
        m.collect(tmp_path/'cache',journal,'private-contact@example.org')
    state=json.loads(journal.read_text())
    assert state['status']=='failed' and not state['completed']
    assert state['current_quarter']=='2009q2'
    assert calls==['2009q2']


def test_disk_reserve_stops_before_download(tmp_path,monkeypatch):
    m=module()
    from types import SimpleNamespace
    monkeypatch.setattr(m.shutil,'disk_usage',lambda root:SimpleNamespace(free=1))
    journal=tmp_path/'state.json'
    with pytest.raises(ValueError,match='free-disk reserve'):
        m.collect(tmp_path/'cache',journal,'private-contact@example.org')
    assert json.loads(journal.read_text())['status']=='failed'

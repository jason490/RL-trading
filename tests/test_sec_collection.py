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


@pytest.mark.parametrize('limit_name,limit,exception', [
    ('DEADLINE_SECONDS',139,TimeoutError),
    ('MAX_BYTES',255_999_999,ValueError),
])
def test_reserve_limits_stop_before_any_request(tmp_path,monkeypatch,limit_name,limit,exception):
    m=module()
    monkeypatch.setattr(m,limit_name,limit)
    calls=[]
    monkeypatch.setattr(m,'download_archive',lambda *args:calls.append(args))
    journal=tmp_path/'state.json'
    with pytest.raises(exception):
        m.collect(tmp_path/'cache',journal,'private-contact@example.org')
    assert not calls
    assert json.loads(journal.read_text())['status']=='failed'


def test_master_collection_uses_separate_downloader_protocol_and_limits(tmp_path,monkeypatch):
    m=module()
    calls=[]
    def fake(root,quarter,user_agent):
        calls.append(quarter)
        raw=quarter.encode()
        p=root/hashlib.sha256(raw).hexdigest()/'archive.zip'
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(raw)
        return p
    monkeypatch.setattr(m,'download_master_archive',fake)
    state=m.collect(tmp_path/'cache',tmp_path/'state.json','private-contact@example.org',source_kind='master')
    assert state['source_kind']=='master'
    assert state['compressed_byte_limit']==1_000_000_000
    assert len(calls)==59
    assert state['status']=='downloaded_only'


def test_unknown_collection_kind_rejected_before_state_creation(tmp_path):
    m=module()
    with pytest.raises(ValueError,match='source kind'):
        m.collect(tmp_path/'cache',tmp_path/'state.json','private-contact@example.org',source_kind='other')
    assert not (tmp_path/'state.json').exists()

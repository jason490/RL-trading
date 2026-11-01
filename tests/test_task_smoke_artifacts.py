import io
import json
import tarfile

import pytest

from rl_stocks.runtime.colab.jobs import task_smoke_plan
from rl_stocks.runtime.colab.task_smoke_artifacts import validate_task_smoke_archive, verify_directory
from test_task_bundle import bundle


def test_incomplete_archive_and_wrong_dataset_rejected(tmp_path):
    plan = task_smoke_plan('tagged_routed')
    path = tmp_path / 'empty.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        item = tarfile.TarInfo('artifacts/completion.json')
        data = b'{}'
        item.size = len(data)
        archive.addfile(item, io.BytesIO(data))
    with pytest.raises(RuntimeError, match='bounded complete'):
        validate_task_smoke_archive(path, plan, plan['dataset_contract']['identity'])
    with pytest.raises(RuntimeError, match='bound task'):
        validate_task_smoke_archive(path, plan, 'wrong')


def test_claimed_success_with_untrained_physical_model_rejected(tmp_path):
    model, _, _ = bundle(tmp_path)
    plan = task_smoke_plan('tagged_routed')
    run = dict(status='complete', task_role='tagged_routed',
        dataset=plan['dataset_contract']['identity'],
        type_manifest_sha256=plan['type_manifest_sha256'], protocol_sha256=plan['protocol_sha256'],
        actual_transitions=1024, optimizer_steps=dict.fromkeys(('actor', 'critic', 'entropy'), 960),
        development_evaluated=False, test_evaluated=False, smoke_only=True)
    (tmp_path / 'run.json').write_text(json.dumps(run))
    (tmp_path / 'curve.json').write_text(json.dumps([{'steps': 512}, {'steps': 1024}]))
    for name in ('initial.zip', 'checkpoint-512.zip', 'checkpoint-1024.zip'):
        model.save(tmp_path / name)
    with pytest.raises(ValueError, match='optimizer counters'):
        verify_directory(tmp_path, plan)


@pytest.mark.parametrize('unsafe', ['../escape', 'symlink', 'duplicate'])
def test_complete_shaped_archive_rejects_unsafe_members(tmp_path, unsafe):
    from rl_stocks.runtime.colab.task_smoke_artifacts import FILES
    plan = task_smoke_plan('tagged_routed')
    names = {'artifacts/completion.json'} | {'artifacts/task-smoke/' + name for name in FILES}
    names |= {'artifacts/task-smoke/frozen/' + 'f' * 64 + '/' + name
              for name in ('model.zip', 'manifest.json', 'checksums.json', 'requirements.lock')}
    path = tmp_path / 'unsafe.tar'
    with tarfile.open(path, 'w') as archive:
        for name in sorted(names):
            item = tarfile.TarInfo(name)
            item.size = 2
            archive.addfile(item, io.BytesIO(b'{}'))
        item = tarfile.TarInfo('artifacts' if unsafe == 'symlink' else
                               'artifacts/completion.json' if unsafe == 'duplicate' else unsafe)
        if unsafe == 'symlink':
            item.type = tarfile.SYMTYPE
            item.linkname = '/tmp'
        archive.addfile(item)
    with pytest.raises(RuntimeError, match='archive'):
        validate_task_smoke_archive(path, plan, plan['dataset_contract']['identity'])


def test_report_json_sidecar_is_required_and_allowlisted(tmp_path, monkeypatch):
    import rl_stocks.runtime.colab.task_smoke_artifacts as verifier
    plan = task_smoke_plan('masked_shared')
    completion = dict(plan=plan, dataset=plan['dataset_contract']['identity'], gpu=None,
                      python=[3, 13, 18], source_git={'revision': 'a' * 40, 'dirty': ''},
                      source_hash='b' * 64)
    monkeypatch.setattr(verifier, 'verify_directory', lambda *args: dict(
        repository=completion['source_git'], source=completion['source_hash'], task_role='masked_shared'))
    assert 'final-training.json' in verifier.FILES
    names = {'artifacts/task-smoke/' + name for name in verifier.FILES}
    names |= {'artifacts/task-smoke/frozen/' + 'f' * 64 + '/' + name
              for name in ('model.zip', 'manifest.json', 'checksums.json', 'requirements.lock')}
    for omit in (False, True):
        path = tmp_path / f'report-{omit}.tar'
        with tarfile.open(path, 'w') as archive:
            for name in sorted(names | {'artifacts/completion.json'}):
                if omit and name.endswith('/final-training.json'):
                    continue
                data = json.dumps(completion).encode() if name.endswith('/completion.json') else b'{}'
                item = tarfile.TarInfo(name)
                item.size = len(data)
                archive.addfile(item, io.BytesIO(data))
        if omit:
            with pytest.raises(RuntimeError, match='bounded complete'):
                verifier.validate_task_smoke_archive(path, plan, completion['dataset'])
        else:
            assert verifier.validate_task_smoke_archive(path, plan, completion['dataset'])[0]['status'] == 'complete'

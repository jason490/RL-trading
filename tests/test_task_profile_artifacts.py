import io
import json
import tarfile

import pytest

from rl_stocks.runtime.colab.jobs import task_profile_plan
from rl_stocks.runtime.colab.task_profile_artifacts import validate_task_profile_archive, verify_profile_directory
from test_task_bundle import bundle


def test_claimed_profile_with_untrained_model_is_rejected(tmp_path):
    plan = task_profile_plan()
    completion = dict(source_git={'revision': 'a' * 40, 'dirty': ''}, source_hash='b' * 64)
    metadata = dict(task_role='tagged_routed', protocol_sha256=plan['protocol_sha256'],
        dataset=plan['dataset_contract']['identity'], type_manifest_sha256=plan['type_manifest_sha256'],
        repository=completion['source_git'], source=completion['source_hash'], profile_only=True,
        seed=11, development_evaluated=False, test_evaluated=False, performance_promotion=False)
    record = dict(metadata, status='complete', actual_transitions=4096,
        optimizer_steps=dict.fromkeys(('actor', 'critic', 'entropy'), 3096),
        performance_scores_exported=False, holding_path_sha256='c' * 64,
        learning_seconds=10., evaluation_seconds=20., inclusive_elapsed_seconds=31.)
    (tmp_path / 'input.json').write_text(json.dumps(metadata))
    (tmp_path / 'profile.json').write_text(json.dumps(record))
    model, _, _ = bundle(tmp_path)
    model.batch_size = 256
    model.learning_starts = 1000
    for name in ('initial.zip', 'profile-final.zip'):
        model.save(tmp_path / name)
    with pytest.raises(ValueError, match='optimizer counters'):
        verify_profile_directory(tmp_path, plan, 'tagged_routed', completion)
    record['learning_seconds'] = float('nan')
    (tmp_path / 'profile.json').write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match='measured profile'):
        verify_profile_directory(tmp_path, plan, 'tagged_routed', completion)


def test_profile_archive_requires_complete_roles(tmp_path):
    plan = task_profile_plan()
    path = tmp_path / 'incomplete.tar'
    with tarfile.open(path, 'w') as archive:
        item = tarfile.TarInfo('artifacts/completion.json')
        item.size = 2
        archive.addfile(item, io.BytesIO(b'{}'))
    with pytest.raises(RuntimeError, match='complete bounded'):
        validate_task_profile_archive(path, plan, plan['dataset_contract']['identity'])

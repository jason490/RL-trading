import json
import tarfile

import pytest

from test_recurrent_environment_verify import directory_fixture, no_fitting  # noqa: F401
from rl_stocks.core.contracts import canonical
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.runtime.colab.jobs import recurrent_environment_plan
from rl_stocks.training import recurrent_environment_worker as worker


def packed(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    folder = root / "recurrent-loop"
    directory_fixture(folder, monkeypatch)
    plan = recurrent_environment_plan()
    (root / "completion.json").write_bytes(
        canonical(
            dict(
                plan=plan,
                dataset=worker.INPUT_ID,
                python=[3, 13, 16],
                gpu=None,
                source_git={"revision": "a" * 40, "dirty": ""},
                source_hash="b" * 64,
            )
        )
    )
    archive = tmp_path / "owned.tar"
    with tarfile.open(archive, "w") as stream:
        stream.add(root, arcname="artifacts")
    return archive, root, plan


def test_physical_numerical_archive_complete_and_allowlisted(tmp_path, monkeypatch):
    archive, root, plan = packed(tmp_path, monkeypatch)
    result = validate_archive(archive, plan, worker.INPUT_ID)
    assert result[0]["actual_steps"] == 128 and result[0]["numerical_artifacts_verified"]
    (root / "unreviewed.txt").write_text("excluded")
    with tarfile.open(archive, "w") as stream:
        stream.add(root, arcname="artifacts")
    with pytest.raises(RuntimeError, match="allowlisted"):
        validate_archive(archive, plan, worker.INPUT_ID)


def test_bad_environment_count_cannot_pass_archive(tmp_path, monkeypatch):
    archive, root, plan = packed(tmp_path, monkeypatch)
    record = root/'recurrent-loop/run.json'
    value = json.loads(record.read_text())
    value['actual_transitions'] = 129
    record.write_text(json.dumps(value))
    with tarfile.open(archive,'w') as stream:
        stream.add(root,arcname='artifacts')
    with pytest.raises(ValueError):
        validate_archive(archive,plan,worker.INPUT_ID)

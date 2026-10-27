import json
import tarfile

import pytest

from test_recurrent_sac_smoke_verify import directory_fixture, no_fitting  # noqa: F401
from rl_stocks.core.contracts import canonical
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.runtime.colab.jobs import recurrent_sac_smoke_plan
from rl_stocks.training import recurrent_sac_smoke as smoke
from rl_stocks.runtime.colab import recurrent_sac_smoke_artifacts as artifacts


def packed(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    folder = root / "recurrent-smoke"
    payload = directory_fixture(folder, monkeypatch)
    (folder / "input.json").write_bytes(canonical(payload))
    monkeypatch.setattr(artifacts, 'protocol', smoke.protocol)
    plan = recurrent_sac_smoke_plan()
    (root / "completion.json").write_bytes(
        canonical(
            dict(
                plan=plan,
                dataset=smoke.protocol()['input_identity'],
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
    result = validate_archive(archive, plan, smoke.protocol()['input_identity'])
    assert result[0]["actual_steps"] == 8 and result[0]["numerical_artifacts_verified"]
    (root / "unreviewed.txt").write_text("excluded")
    with tarfile.open(archive, "w") as stream:
        stream.add(root, arcname="artifacts")
    with pytest.raises(RuntimeError, match="allowlisted"):
        validate_archive(archive, plan, smoke.protocol()['input_identity'])


def test_bad_learned_summary_cannot_pass_archive(tmp_path, monkeypatch):
    archive, root, plan = packed(tmp_path, monkeypatch)
    record = root / 'recurrent-smoke/summary.json'
    value = json.loads(record.read_text())
    value['adam_steps'] = 25
    record.write_text(json.dumps(value))
    with tarfile.open(archive, 'w') as stream:
        stream.add(root, arcname='artifacts')
    with pytest.raises(ValueError):
        validate_archive(archive, plan, smoke.protocol()['input_identity'])

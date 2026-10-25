"""Public GAE bundles preserve exact checked arrays and exclude private inputs."""

import importlib.util
import json
import tarfile

import pytest

from test_gae_artifacts import archive
from rl_stocks.reporting.gae_public import public_evidence


@pytest.mark.parametrize("arm", ["control", "mc_trace"])
def test_checked_gae_arrays_initializer_and_labels_published(tmp_path, arm):
    archive(tmp_path, arm=arm)
    root = tmp_path / "artifacts"
    run = root / "runs/fixture"
    metadata = json.loads((run / "metadata.json").read_text())
    expected = public_evidence(run, metadata)
    (run / ".env").write_text("private")
    (run / "replay.pkl").write_bytes(b"private replay")
    report = {"fixture": dict(
        rows=[dict(session="2021-01-04", equity=100000, price=987654)],
        baselines=dict(buy_hold=dict(rows=[dict(equity=100000)])),
        metrics=dict(net_return=0),
    )}
    (run / "selected-validation.json").write_text(json.dumps(report))
    spec = importlib.util.spec_from_file_location("gae_package", "scripts/package_research.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "public"
    package = module.package([root], output)
    with tarfile.open(package) as stream:
        for path in expected:
            name = "runs/fixture/" + str(path.relative_to(run))
            assert stream.extractfile(name).read() == path.read_bytes()
        assert not any("replay" in n or ".env" in n for n in stream.getnames())
    assert arm in (output / "validation.html").read_text()
    assert "987654" not in (output / "validation/fixture.html").read_text()
    assert len(expected) == 6


@pytest.mark.parametrize("change", ["extra", "symlink", "failure", "missing", "corrupt"])
def test_unreviewed_or_invalid_gae_evidence_refused(tmp_path, change):
    archive(tmp_path)
    run = tmp_path / "artifacts/runs/fixture"
    metadata = json.loads((run / "metadata.json").read_text())
    first = run / "gae-rollouts/rollout-000128.npz"
    if change == "extra":
        (first.parent / "private.csv").write_text("not allowed")
    elif change == "symlink":
        first.unlink()
        first.symlink_to(run / "metadata.json")
    elif change == "failure":
        (run / "gae-rollout-failure.json").write_text("{}")
    elif change == "missing":
        first.unlink()
    else:
        first.write_bytes(b"bad array")
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        public_evidence(run, metadata)

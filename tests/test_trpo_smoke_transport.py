import importlib.util
import io
from pathlib import Path
import tarfile

import pytest

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.runtime.colab import trpo_smoke_artifacts as transport


def archive(path, names):
    with tarfile.open(path, "w:gz") as stream:
        for name, kind, data in names:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(data)
            stream.addfile(member, io.BytesIO(data))


@pytest.mark.parametrize("names", [
    [("../escape", tarfile.REGTYPE, b"x")],
    [("artifacts/link", tarfile.SYMTYPE, b"")],
    [("artifacts/x", tarfile.REGTYPE, b"x")] * 2,
    [("unrelated/x", tarfile.REGTYPE, b"x")],
])
def test_container_rejects_unsafe_members_before_extraction(tmp_path, names):
    source = tmp_path / "archive.tar.gz"
    archive(source, names)
    with pytest.raises(ValueError):
        transport.extract_owned(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_transport_failure_and_retained_regular_archive(tmp_path):
    source = tmp_path / "archive.tar.gz"
    archive(source, [("artifacts/failure.txt", tarfile.REGTYPE, b"remote failure")])
    with pytest.raises(RuntimeError, match="failed"):
        transport.extract_owned(source, tmp_path / "out")
    archive(source, [("artifacts/result.json", tarfile.REGTYPE, b"{}")])
    out = transport.extract_owned(source, tmp_path / "out")
    assert (out / "result.json").read_text() == "{}" and source.exists()


def fixture(tmp_path, monkeypatch):
    dataset = {"normalizer": {"means": [0]}, "names": ["return"]}
    source = {"revision": "fixture", "dirty": ""}
    spec = {"dataset": digest(dataset), "normalizer_sha256": digest(dataset["normalizer"]),
            "cases": []}
    monkeypatch.setattr(transport, "protocol", lambda: spec)
    folder = tmp_path / "artifacts"
    (folder / "trpo-smoke").mkdir(parents=True)
    result = {"schema": "trpo-smoke-result-v1", "source": source, "source_hash": "abc",
              "protocol_sha256": transport.PROTOCOL_SHA256, "dataset": spec["dataset"],
              "normalizer_sha256": spec["normalizer_sha256"],
              "normalizer": dataset["normalizer"], "names": dataset["names"],
              "development_evaluated": False, "retired_test_access": False,
              "performance_promotion": False, "training_location": "Colab CPU", "results": []}
    (folder / "trpo-smoke/result.json").write_bytes(canonical(result))
    completion = {"schema": "trpo-smoke-completion-v1", "python": [3, 13, 18],
                  "source": source, "source_hash": "abc",
                  "protocol_sha256": transport.PROTOCOL_SHA256, "dataset": spec["dataset"],
                  "training_location": "Colab CPU", "development_evaluated": False,
                  "retired_test_access": False,
                  "files": {"trpo-smoke/result.json": transport.checksum(folder / "trpo-smoke/result.json")}}
    (folder / "completion.json").write_bytes(canonical(completion))
    return folder, dataset, source, result


def test_root_source_preprocessing_and_manifest(tmp_path, monkeypatch):
    folder, dataset, source, result = fixture(tmp_path, monkeypatch)
    transport.verify_root(folder, source=source, source_hash="abc", dataset=dataset)
    with pytest.raises(ValueError, match="identity"):
        transport.verify_root(folder, source=source, source_hash="changed", dataset=dataset)
    result["normalizer"]["means"] = [9]
    (folder / "trpo-smoke/result.json").write_bytes(canonical(result))
    with pytest.raises(ValueError, match="identity"):
        transport.verify_root(folder, source=source, source_hash="abc", dataset=dataset)


def test_root_rejects_additional_unmanifested_file(tmp_path, monkeypatch):
    folder, dataset, source, _ = fixture(tmp_path, monkeypatch)
    (folder / "trpo-smoke/extra").write_text("unexpected")
    with pytest.raises(ValueError, match="manifest"):
        transport.verify_root(folder, source=source, source_hash="abc", dataset=dataset)


def test_remote_default_kernel_guard_precedes_io(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/remote_trpo_smoke.py"
    spec = importlib.util.spec_from_file_location("remote_trpo_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    with pytest.raises(RuntimeError, match="default Colab Python"):
        module.main()
    monkeypatch.setattr(module.time, "time", lambda: 100)
    with pytest.raises(TimeoutError):
        module.remaining(1000)
    assert module.remaining(2000) == 1000


def test_container_expanded_size_and_member_count_bounds(tmp_path, monkeypatch):
    source = tmp_path / "archive.tar.gz"
    archive(source, [(f"artifacts/{i}", tarfile.REGTYPE, b"") for i in range(101)])
    with pytest.raises(ValueError, match="count/expanded"):
        transport.extract_owned(source, tmp_path / "count-out")
    archive(source, [("artifacts/large", tarfile.REGTYPE, b"a" * 50000)])
    monkeypatch.setattr(transport, "MAXIMUM_BYTES", 1000)
    assert source.stat().st_size < 1000
    with pytest.raises(ValueError, match="count/expanded"):
        transport.extract_owned(source, tmp_path / "size-out")
    assert not (tmp_path / "size-out").exists()

import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

from rl_stocks.reporting.retained_reports import run_reports
from rl_stocks.runtime.colab.source_archive import source_member


@pytest.mark.parametrize("name", ["critic", "crossq", "gae", "treasury", "vix"])
def test_reporting_entry_points_share_retained_loader(name):
    path = Path(__file__).resolve().parents[1] / f"scripts/report_{name}_study.py"
    spec = importlib.util.spec_from_file_location(f"report_{name}_cleanup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.run_reports is run_reports


def test_retained_reports_require_one_run_and_preserve_both_phases(tmp_path):
    with pytest.raises(ValueError, match="one retained run"):
        run_reports(tmp_path)
    run = tmp_path / "runs/first"
    run.mkdir(parents=True)
    for name, value in (("metadata.json", {"seed": 11}),
                        ("selected-validation.json", {"return": .2}),
                        ("validation.json", {"return": .1})):
        (run / name).write_text(json.dumps(value))
    directory, metadata, phases = run_reports(tmp_path)
    assert directory == run and metadata == {"seed": 11}
    assert phases == {"selected": {"return": .2}, "final": {"return": .1}}
    second = tmp_path / "runs/second"
    second.mkdir()
    (second / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="one retained run"):
        run_reports(tmp_path)


def test_source_archive_omits_cache_but_preserves_python_sources(tmp_path):
    root = tmp_path / "src/pkg"
    (root / "__pycache__").mkdir(parents=True)
    for name in ("main.py", "orphan.pyc", "legacy.pyo", "__pycache__/main.cpython-313.pyc"):
        (root / name).write_text("fixture")
    path = tmp_path / "source.tar.gz"
    with tarfile.open(path, "w:gz") as stream:
        stream.add(tmp_path / "src", arcname="src", filter=source_member)
    with tarfile.open(path) as stream:
        names = stream.getnames()
    assert "src/pkg/main.py" in names
    assert not any("__pycache__" in n or n.endswith((".pyc", ".pyo")) for n in names)

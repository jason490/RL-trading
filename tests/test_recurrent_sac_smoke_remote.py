"""Remote dispatch only, with numerical execution replaced by a read-only stub."""

import importlib.util
import json
import os
from pathlib import Path

import torch

from test_sac_update_input import fixture
from rl_stocks.core.contracts import digest
from rl_stocks.runtime.colab.jobs import recurrent_sac_smoke_plan
from rl_stocks.training import recurrent_sac_smoke_worker as worker
from rl_stocks.training import recurrent_sac_smoke as smoke


def test_remote_routes_probe_without_generic_trainer_or_clearing_stop(tmp_path, monkeypatch):
    payload = fixture()
    spec = smoke.protocol()
    spec['input_identity'] = digest(payload)
    monkeypatch.setattr(smoke, 'protocol', lambda value=spec: value)
    plan = recurrent_sac_smoke_plan()
    spec = importlib.util.spec_from_file_location(
        "remote_probe", Path(__file__).parents[1] / "scripts/remote_train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "root", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for key in (
        "UV_CACHE_DIR",
        "MPLCONFIGDIR",
        "MPLBACKEND",
        "PYTHONPATH",
        "RL_CHECKPOINT_ARCHIVE",
    ):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    (tmp_path / "dataset.json").write_text(json.dumps(payload))
    (tmp_path / "job-plan.json").write_text(json.dumps(plan))
    (tmp_path / "source-git.json").write_text(json.dumps(dict(revision="a" * 40, dirty="")))
    (tmp_path / "artifacts").mkdir()
    stop = tmp_path / "crossq-integrity-stop.json"
    stop.write_text("preserved")
    called = []
    monkeypatch.setattr(
        worker, "execute_colab", lambda data, output: called.append((digest(data), output))
    )
    module.main()
    assert called == [(smoke.protocol()['input_identity'], tmp_path / "artifacts/recurrent-smoke")]
    completion = json.loads((tmp_path / "artifacts/completion.json").read_text())
    assert completion["plan"] == plan and completion["dataset"] == smoke.protocol()['input_identity']
    assert completion["gpu"] is None and completion["source_git"]["dirty"] == ""
    assert stop.read_text() == "preserved"

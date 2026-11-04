from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from rl_stocks.core.contracts import digest
from rl_stocks.inference.canonical_actor import encode32
from rl_stocks.inference.canonical_tanh_probe import execute, verify
from rl_stocks.runtime.colab.tanh_probe import TanhProbeJob, worker_script
from rl_stocks.runtime.colab.budget import BudgetStop
from test_canonical_tanh import fixture
from test_canonical_probe import FakeProbe


@pytest.fixture(scope="module")
def probe():
    inputs = {
        "actor": fixture(),
        "inputs_hex": [[encode32(((i + j) % 17 - 8) / 16) for j in range(21)] for i in range(128)],
    }
    expected = execute(inputs)
    expected["native_packages_imported"] = []
    return inputs, expected


def test_isolated_worker_matches_exact_reference_without_native_packages(tmp_path, probe):
    inputs, expected = probe
    out = tmp_path / "result.json"
    script = tmp_path / "worker.py"
    script.write_text(worker_script(Path(__file__).resolve().parents[1], inputs, str(out)))
    subprocess.run([sys.executable, str(script)], check=True, capture_output=True, timeout=30)
    actual = json.loads(out.read_text())
    assert verify(actual, expected)
    assert len(actual["output_hex"]) == 128 and not actual["native_packages_imported"]


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v["output_hex"][0].__setitem__(0, "00000000"),
        lambda v: v.update(native_packages_imported=["torch"]),
        lambda v: v.update(python_major_minor=[3, 14]),
        lambda v: v.update(elapsed_seconds=float("nan")),
        lambda v: v.update(training_performed=True),
    ],
)
def test_rejects_different_bits_runtime_or_flags(probe, change):
    _, expected = probe
    actual = deepcopy(expected)
    change(actual)
    with pytest.raises(ValueError):
        verify(actual, expected)


class FakeTanh(TanhProbeJob, FakeProbe):
    pass


@pytest.mark.parametrize("failure", [None, "exec", "meter", "download", "budget", "interrupt"])
def test_reuses_metered_owned_cleanup_without_retry(tmp_path, probe, monkeypatch, failure):
    from rl_stocks.runtime.colab import tanh_probe, reference_probe

    inputs, expected = probe
    monkeypatch.setattr(
        reference_probe, "repository_state", lambda p: {"dirty": "", "revision": "a" * 40}
    )
    monkeypatch.setattr(tanh_probe, "worker_script", lambda *a: "# bounded fake worker")
    monkeypatch.setattr(
        tanh_probe,
        "probe_spec",
        lambda root: {
            "actor_sha256": inputs["actor"]["sha256"],
            "source_model_sha256": "a" * 64,
            "inputs_sha256": digest(inputs["inputs_hex"]),
            "outputs_sha256": expected["output_sha256"],
            "maximum_probe_units": 0.05,
            "deadline_seconds": 1200,
            "teardown_reserve_seconds": 360,
            "execution_seconds": 240,
        },
    )
    job = FakeTanh(tmp_path, expected, failure)
    claim = tmp_path / ".runtime/tanh-claim.json"
    if failure:
        with pytest.raises((RuntimeError, BudgetStop, KeyboardInterrupt)):
            job.run_probe(inputs, expected, claim)
    else:
        assert job.run_probe(inputs, expected, claim) == expected
    assert job.stops and not job.remote_active and job.ledger.active["status"] == "stopped"
    assert job.calls.count("new") == 1
    with pytest.raises(FileExistsError):
        job.run_probe(inputs, expected, claim)
    assert job.calls.count("new") == 1

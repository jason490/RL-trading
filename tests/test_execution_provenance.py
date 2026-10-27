"""Read-only actual installed-backend capture; no learning or allocation."""
import json

import torch

from rl_stocks.core.contracts import canonical
from rl_stocks.core.execution_provenance import execution_backend
from rl_stocks.training.distributional_critics import rng_record


def test_actual_capture_preserves_rng_flags_and_avoids_gpu_inventory(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU provenance must not query GPU names or change backend settings')

    monkeypatch.setattr(torch.cuda, 'get_device_name', forbidden)
    monkeypatch.setattr(torch, 'set_num_threads', forbidden)
    monkeypatch.setattr(torch, 'set_num_interop_threads', forbidden)
    monkeypatch.setattr(torch, 'use_deterministic_algorithms', forbidden)
    before = canonical(rng_record())
    first = execution_backend('cpu')
    second = execution_backend(torch.device('cpu'))
    assert first == second and canonical(rng_record()) == before
    assert first['torch_intraop_threads'] == torch.get_num_threads()
    assert first['torch_interop_threads'] == torch.get_num_interop_threads()
    assert first['cpu_capability'] == torch.backends.cpu.get_cpu_capability()
    assert first['deterministic_algorithms'] == torch.are_deterministic_algorithms_enabled()
    assert first['gpu_name'] is None
    assert json.loads(json.dumps(first)) == first


def test_thread_environment_is_allowlisted_and_never_contains_arbitrary_values(monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '2')
    monkeypatch.setenv('MKL_NUM_THREADS', 'PRIVATE_FIXTURE_VALUE')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '9' * 100)
    monkeypatch.delenv('NUMEXPR_NUM_THREADS', raising=False)
    monkeypatch.setenv('PRIVATE_FIXTURE_KEY', 'EXCLUDED')
    value = execution_backend('cpu')
    assert value['thread_environment'] == dict(OMP_NUM_THREADS=2, MKL_NUM_THREADS='non-numeric',
                                              OPENBLAS_NUM_THREADS='non-numeric', NUMEXPR_NUM_THREADS=None)
    assert 'PRIVATE_FIXTURE' not in json.dumps(value)
    assert 'EXCLUDED' not in json.dumps(value)

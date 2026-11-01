import os

import torch

from rl_stocks.training.task_backend import task_execution_backend


def test_task_backend_reads_selectors_without_mutating_rng_or_environment(monkeypatch):
    monkeypatch.setenv('MKL_CBWR', 'AVX2')
    monkeypatch.setenv('MKL_ENABLE_INSTRUCTIONS', 'private-unrecognized-value')
    before_env = dict(os.environ)
    before_rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    record = task_execution_backend('cpu')
    assert record['cpu_math_environment']['MKL_CBWR'] == 'AVX2'
    assert record['cpu_math_environment']['MKL_ENABLE_INSTRUCTIONS'] == 'unrecognized'
    assert 'private-unrecognized-value' not in str(record)
    assert len(record['torch_build_config_sha256']) == 64
    assert torch.equal(before_rng, torch.get_rng_state())
    assert threads == torch.get_num_threads()
    assert dict(os.environ) == before_env

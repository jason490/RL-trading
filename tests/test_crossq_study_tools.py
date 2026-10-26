"""Research tooling boundaries and actual untrained shared-start comparisons."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from test_crossq_state import model, no_optimization  # noqa: F401
from rl_stocks.core.contracts import canonical
from rl_stocks.runtime.colab.jobs import crossq_plan
from rl_stocks.training.crossq_initialization import apply_initialization, validate_initial_comparison
from rl_stocks.training.crossq_state import model_record


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def starts():
    result = {}
    for arm in ('sac', 'crossq'):
        candidate = model(arm)
        apply_initialization(candidate, arm, smoke=True)
        result[arm] = model_record(candidate)
    return result


def test_observed_initial_records_match_pinned_linear_and_rng():
    assert validate_initial_comparison(starts(), True) == dict(
        shared_linear_tensors_verified=True, sampling_rng_identical=True)


@pytest.mark.parametrize('change', ['missing', 'linear', 'rng', 'seed', 'optimizer'])
def test_comparison_rejects_nominal_or_unmatched_starts(change):
    records = deepcopy(starts())
    if change == 'missing':
        records.pop('sac')
    elif change == 'linear':
        records['crossq']['linear_tensor_sha256'] = '0' * 64
    elif change == 'rng':
        records['crossq']['rng'] = {}
    elif change == 'seed':
        records['crossq']['seed'] = 23
    else:
        records['crossq']['optimizers']['actor']['state_count'] = 1
    with pytest.raises((ValueError, KeyError)):
        validate_initial_comparison(records, True)


def test_builder_outputs_only_two_exact_smokes(tmp_path, monkeypatch):
    target = tmp_path / 'jobs.json'
    monkeypatch.setattr(sys, 'argv', ['build_crossq_jobs', '--smoke', '--output', str(target)])
    script('build_crossq_jobs').main()
    jobs = json.loads(target.read_text())
    assert len(jobs) == 2
    for task, arm in zip(jobs, ('sac', 'crossq'), strict=True):
        assert task['id'] == f'crossq-{arm}-11-smoke'
        assert canonical(task['plan']) == canonical(crossq_plan(arm, smoke=True))


def test_confirmation_builder_requires_passing_screen(tmp_path, monkeypatch):
    target = tmp_path / 'jobs.json'
    monkeypatch.setattr(sys, 'argv', ['build_crossq_jobs', '--seed', '23', '--output', str(target)])
    with pytest.raises(SystemExit):
        script('build_crossq_jobs').main()
    assert not target.exists()


def test_report_rejects_incomplete_fleet_before_dataset_or_deserialization(tmp_path, monkeypatch):
    module = script('report_crossq_study')
    monkeypatch.setattr(module, 'make_runtime_profile', lambda root: {})
    state = tmp_path / 'state.json'
    state.write_text(json.dumps(dict(status='running', completed={}, cleanup_verified=False)))
    source = tmp_path / 'source.json'
    source.write_text('{}')
    with pytest.raises(ValueError, match='completed owned-cleaned'):
        module.report(state, source, tmp_path)

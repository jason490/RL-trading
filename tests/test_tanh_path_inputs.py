import hashlib
from types import SimpleNamespace

import pytest

from rl_stocks.runtime.colab import tanh_path_inputs as inputs


def test_declared_imports_include_function_local_and_relative_sources(tmp_path):
    package = tmp_path / 'src/rl_stocks'
    package.mkdir(parents=True)
    (package / 'entry.py').write_text('from . import common\ndef f():\n import rl_stocks.lazy\n')
    (package / 'common.py').write_text('import math\n')
    (package / 'lazy.py').write_text('value=1\n')
    actual = inputs.numerical_imports(tmp_path, 'rl_stocks.entry')
    assert set(actual) == {'src/rl_stocks/entry.py', 'src/rl_stocks/common.py', 'src/rl_stocks/lazy.py'}
    assert all(len(value) == 64 for value in actual.values())


def test_baseline_source_review_rejects_changed_calculations(tmp_path, monkeypatch):
    original = b'original source'
    current = {'src/rl_stocks/calculate.py': hashlib.sha256(original).hexdigest()}
    monkeypatch.setattr(inputs, 'numerical_imports', lambda root: current)
    calls = []
    def show(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=original)
    monkeypatch.setattr(inputs.subprocess, 'run', show)
    assert inputs.reviewed_numerical_sources(tmp_path) == current
    assert calls[0] == ['git', 'show', inputs.BASELINE_REVISION + ':src/rl_stocks/calculate.py']
    current['src/rl_stocks/calculate.py'] = hashlib.sha256(b'changed').hexdigest()
    with pytest.raises(ValueError, match='calculation source changed'):
        inputs.reviewed_numerical_sources(tmp_path)


def test_incomplete_local_result_rejects_before_restoration(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('incomplete panel reached full restoration')
    monkeypatch.setattr(inputs, 'verify_directory', forbidden)
    with pytest.raises(ValueError, match='complete local'):
        inputs.admit_local(tmp_path, {})

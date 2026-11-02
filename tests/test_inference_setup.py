import subprocess
import sys

import pytest

from rl_stocks.runtime.colab.inference_setup import SetupFailure, checked, create_environment


def test_actual_venv_without_pip_preserves_current_interpreter(tmp_path):
    target = tmp_path / 'isolated'
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(target)],
                   check=True, timeout=30)
    code = 'import sys,importlib.util;print(sys.version);assert importlib.util.find_spec("pip") is None'
    result = subprocess.run([str(target / 'bin/python'), '-I', '-c', code], check=True,
                            capture_output=True, text=True, timeout=30)
    assert result.stdout.strip() == sys.version


def test_install_targets_isolated_python_without_ensurepip(tmp_path):
    calls = []
    def execute(stage, command, timeout):
        calls.append((stage, command))
        return '--python' if stage == 'kernel-pip-preflight' else ''
    target = tmp_path / 'venv'
    python = create_environment(sys.executable, target, tmp_path / 'inference.lock', execute=execute)
    assert python == str(target / 'bin/python')
    assert calls[1][1] == [sys.executable, '-m', 'venv', '--without-pip', str(target)]
    install = calls[2][1]
    assert install[:5] == [sys.executable, '-m', 'pip', '--python', python]
    assert '--require-hashes' in install and '--no-deps' in install


def test_missing_host_pip_fails_before_creating_environment(tmp_path):
    calls = []
    def missing(stage, command, timeout):
        calls.append(stage)
        raise SetupFailure(stage, 1, stderr='No module named pip')
    with pytest.raises(SetupFailure) as failure:
        create_environment(sys.executable, tmp_path / 'venv', tmp_path / 'lock', execute=missing)
    assert calls == ['kernel-pip-preflight']
    assert 'No module named pip' in failure.value.record['stderr']
    assert not (tmp_path / 'venv').exists()


def test_child_failure_captures_actual_stderr_and_redacts_urls():
    code = 'import sys;print("https://user:secret@example.com/a",file=sys.stderr);print("ensurepip unavailable",file=sys.stderr);sys.exit(3)'
    with pytest.raises(SetupFailure) as failure:
        checked('fixture-child', [sys.executable, '-c', code], 30)
    record = failure.value.record
    assert record['returncode'] == 3 and record['stage'] == 'fixture-child'
    assert 'ensurepip unavailable' in record['stderr']
    assert 'secret' not in record['stderr'] and '[URL omitted]' in record['stderr']

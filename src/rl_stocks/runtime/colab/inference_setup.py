"""Create an inference venv using the kernel interpreter and its existing pip.

Do not rely on OS distributions bundling ensurepip with their kernel Python.
All package installation targets the isolated venv, never the base environment.
"""
import re
import subprocess
from pathlib import Path


class SetupFailure(RuntimeError):
    def __init__(self, stage, returncode, stdout='', stderr=''):
        super().__init__(f'{stage} failed (return code {returncode})')
        self.record = {'stage': stage, 'returncode': returncode,
                       'stdout': _safe(stdout), 'stderr': _safe(stderr)}


def _safe(value):
    if isinstance(value, bytes):
        value = value.decode(errors='replace')
    return re.sub(r'https?://\S+', '[URL omitted]', value or '')[-20000:]


def checked(stage, command, timeout):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise SetupFailure(stage, 'timeout', error.stdout, error.stderr) from error
    if result.returncode:
        raise SetupFailure(stage, result.returncode, result.stdout, result.stderr)
    return result.stdout


def create_environment(kernel_python, target, lock, *, execute=checked):
    target = Path(target)
    if target.exists():
        raise ValueError('fresh inference environment required')
    help_text = execute('kernel-pip-preflight', [str(kernel_python), '-m', 'pip', '--help'], 30)
    if '--python' not in help_text:
        raise SetupFailure('kernel-pip-preflight', 'unsupported', stderr='Existing kernel pip lacks --python')
    execute('create-venv-without-pip',
            [str(kernel_python), '-m', 'venv', '--without-pip', str(target)], 120)
    python = str(target / 'bin/python')
    execute('install-inference-dependencies', [str(kernel_python), '-m', 'pip', '--python', python,
            'install', '--disable-pip-version-check', '--only-binary=:all:', '--require-hashes',
            '--no-deps', '-r', str(lock)], 300)
    return python

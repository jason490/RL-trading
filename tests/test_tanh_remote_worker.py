import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/remote_tanh_paths.py'
    spec = importlib.util.spec_from_file_location('remote_tanh_path_fixture', path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_remote_worker_blocks_local_before_dataset_io(monkeypatch, tmp_path):
    worker = module()
    monkeypatch.delenv('COLAB_RELEASE_TAG', raising=False)
    with pytest.raises(RuntimeError, match='default Colab Python'):
        worker.main()
    with pytest.raises(RuntimeError):
        worker.replay(tmp_path)
    assert not list(tmp_path.iterdir())


def test_notebook_args_do_not_replace_explicit_subprocess_worker(monkeypatch):
    worker = module()
    monkeypatch.setenv('COLAB_RELEASE_TAG', 'fixture')
    monkeypatch.setenv('RL_ALLOCATION_DEADLINE', '9999999999')
    monkeypatch.setattr(worker.sys, 'argv', ['ipykernel_launcher.py', '-f', 'kernel.json'])
    class Root:
        def mkdir(self, *, exist_ok):
            assert not exist_ok
            raise RuntimeError('owned bootstrap branch selected')
    monkeypatch.setattr(worker, 'Path', lambda value: Root())
    with pytest.raises(RuntimeError, match='bootstrap branch'):
        worker.main()

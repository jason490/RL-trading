import pytest
from rl_stocks.training.task_screen_worker import execute_screen_colab


def test_screen_rejects_local_optimization_before_outputs(tmp_path):
    output = tmp_path / 'artifacts' / 'screen'
    with pytest.raises(RuntimeError, match='owned Colab CPU'):
        execute_screen_colab(None, None, output, task_role='tagged_routed')
    assert not output.exists()

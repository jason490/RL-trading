import pytest
from rl_stocks.training.task_profile_worker import execute_profile_colab


def test_profile_rejects_local_execution_before_creating_output(tmp_path):
    output = tmp_path / 'artifacts' / 'profile'
    with pytest.raises(RuntimeError, match='owned Colab CPU'):
        execute_profile_colab(None, None, output, task_role='tagged_routed')
    assert not output.exists()

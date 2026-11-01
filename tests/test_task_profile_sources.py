import subprocess

import pytest

from rl_stocks.training.task_profile_sources import verify_profile_sources


def test_profile_source_binding_rejects_changes_and_added_compute_code(tmp_path):
    path = tmp_path / 'src/rl_stocks/rl/sac/config.py'
    path.parent.mkdir(parents=True)
    path.write_text('learning_rate = .0003\n')
    (tmp_path / 'requirements.lock').write_text('fixed dependencies\n')
    def git(*args):
        return subprocess.run(['git', *args], cwd=tmp_path, check=True, capture_output=True,
                              text=True).stdout.strip()
    git('init')
    git('add', '.')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'fixture')
    revision = git('rev-parse', 'HEAD')
    assert len(verify_profile_sources(tmp_path, revision)) == 2
    path.write_text('learning_rate = .01\n')
    with pytest.raises(ValueError, match='source changed'):
        verify_profile_sources(tmp_path, revision)
    path.write_text('learning_rate = .0003\n')
    (path.parent / 'extra.py').write_text('new = True\n')
    with pytest.raises(ValueError, match='inventory changed'):
        verify_profile_sources(tmp_path, revision)

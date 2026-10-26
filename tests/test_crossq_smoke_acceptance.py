"""Incomplete/recovered fleet states never become learned smoke evidence."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('accept_crossq', Path(__file__).parents[1] / 'scripts/accept_crossq_smokes.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('change', ['running', 'cleanup', 'missing_arm', 'extra_session', 'unfinished'])
def test_reject_incomplete_or_unowned_smoke_state(tmp_path, change):
    tasks = ['crossq-sac-11-smoke', 'crossq-crossq-11-smoke']
    state = dict(status='complete', cleanup_verified=True, completed={t: {} for t in tasks},
                 sessions={str(i): dict(task=t, phase='done') for i, t in enumerate(tasks)})
    if change == 'running':
        state['status'] = 'running'
    elif change == 'cleanup':
        state['cleanup_verified'] = False
    elif change == 'missing_arm':
        state['completed'].pop(tasks[0])
    elif change == 'extra_session':
        state['sessions']['other'] = dict(task='unrelated', phase='done')
    else:
        state['sessions']['0']['phase'] = 'stopping'
    state_path, source_path = tmp_path / 'state.json', tmp_path / 'source.json'
    state_path.write_text(json.dumps(state))
    source_path.write_text('{}')
    with pytest.raises(ValueError):
        module.accept(state_path, source_path, tmp_path)

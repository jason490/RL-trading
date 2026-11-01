from copy import deepcopy

import pytest

from rl_stocks.training.task_selection import select_training_checkpoint


def history():
    return [{'steps': step, 'phase': 'training-window', 'score': .1,
             'holding_path_sha256': 'a' * 64} for step in range(10000, 100001, 10000)]


def test_earliest_training_max_keeps_final_without_granting_development():
    points = history()
    points[2]['score'] = points[5]['score'] = .4
    before = deepcopy(points)
    result = select_training_checkpoint(points)
    assert result['selected']['steps'] == 30000
    assert result['final']['steps'] == 100000
    assert not result['development_allowed']
    result['selected']['score'] = 123
    assert points == before


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'development', 'nan', 'holding'])
def test_incomplete_or_contaminated_selection_rejected(change):
    points = history()
    if change == 'missing':
        points.pop()
    elif change == 'duplicate':
        points[1]['steps'] = points[0]['steps']
    elif change == 'development':
        points[-1]['phase'] = 'validation'
    elif change == 'nan':
        points[-1]['score'] = float('nan')
    else:
        points[-1]['holding_path_sha256'] = 'b' * 64
    with pytest.raises(ValueError):
        select_training_checkpoint(points)

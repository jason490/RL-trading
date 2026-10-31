from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest

from rl_stocks.evaluation.feasible_references import FixedReference, protocol
from rl_stocks.evaluation.offpolicy_fit import evaluate_fixed_window
from rl_stocks.risk.decisions import decode


def test_reference_is_unconditional_bounded_and_returns_independent_arrays():
    for name, action in protocol()['actions'].items():
        policy = FixedReference(name)
        first, _ = policy.predict(np.array([1.]))
        second, _ = policy.predict(np.array([-999., 10.]))
        assert first.tolist() == second.tolist() == action and np.max(np.abs(first)) <= 1
        first[0] = .5
        assert policy.predict(None)[0].tolist() == action
        with pytest.raises(FrozenInstanceError):
            policy.action = (0., 0., 0., 0.)
        with pytest.raises(ValueError):
            policy.predict(None, deterministic=False)
    with pytest.raises(ValueError):
        FixedReference('oracle')


def test_boundary_geometry_remains_real_long_and_short_protection():
    for name in ('long', 'short'):
        action = FixedReference(name).predict(None)[0]
        decision = decode(action, 'SPY', '2020-01-02', '2020-01-03', 100., 2., 100000.)
        sign = 1 if name == 'long' else -1
        assert decision.quantity == sign*1000
        assert decision.stop == 100-sign*10
        assert decision.near == 100+sign*16 and decision.far == 100+sign*24
    assert decode(FixedReference('cash').predict(None)[0], 'SPY', 'x', 'y', 100., 2., 100000.).stop is None


def test_actual_cash_path_is_flat_and_training_dates_are_kept():
    from test_training_fit import data
    from test_core import bar
    source = data()
    source['episodes']['train']['bars'] = [asdict(bar(d)) for d in
        ['2017-12-28', '2017-12-29', '2018-01-02', '2020-12-02', '2020-12-03']]
    source.update(names=['fixture'], normalizer=dict(names=['fixture'], means=[0.], scales=[1.]))
    result = evaluate_fixed_window(FixedReference('cash'), source)['train']
    assert result['split'] == 'train' and result['events'] == []
    assert result['metrics']['net_return'] == 0 and result['metrics']['costs'] == 0
    assert [r['session'] for r in result['rows']] == ['2018-01-02', '2020-12-02']


def test_constant_long_still_exits_at_holding_limit():
    from datetime import date, timedelta
    from test_training_fit import data
    from test_core import bar
    source = data()
    day = date(2018, 1, 2)
    sessions = []
    while len(sessions) < 24:
        if day.weekday() < 5:
            sessions.append(day.isoformat())
        day += timedelta(days=1)
    dates = ['2017-12-29']+sessions+['2020-12-02']
    source['episodes']['train'].update(bars=[asdict(bar(d)) for d in dates],
                                      features=[[0.] for _ in dates], atr=[1.]*len(dates))
    source.update(names=['fixture'], normalizer=dict(names=['fixture'], means=[0.], scales=[1.]))
    result = evaluate_fixed_window(FixedReference('long'), source)['train']
    assert any(event['reason'] == 'holding_limit' for event in result['events'])
    assert all(event['holding_sessions'] <= 20 for event in result['events'] if event['closed'])
    assert all(event['holding_sessions'] == 20 for event in result['events']
               if event['reason'] == 'holding_limit')

from copy import deepcopy
from dataclasses import asdict
import pytest
from test_core import bar
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.runner import baselines
from rl_stocks.evaluation.metrics import metrics
from rl_stocks.evaluation.direction import directional_attribution
from rl_stocks.evaluation.report_contract import validate_development_report


def fixture():
    bars=[asdict(bar(day,dividend=.1 if day.endswith('06') else 0.)) for day in ['2021-01-04','2021-01-05','2021-01-06']]
    e=dict(symbol='SPY',split='validation',bars=bars);rows=[dict(session=b['session'],price=b['close'],equity=100000.,previous_equity=100000.,quantity=0,costs=0.,turnover=0.,borrow_cost=0.) for b in bars[1:]]
    r=dict(symbol='SPY',split='validation',rows=rows,events=[],metrics=metrics(rows,[]),baselines=baselines(e,SimulationConfig()),directional=directional_attribution(e,rows,[]))
    return {'SPY-validation':r},{'episodes':{'SPY-validation':e}},SimulationConfig()


def test_complete_saved_development_path_and_actions_validate():
    assert validate_development_report(*fixture())['episodes']==1


@pytest.mark.parametrize('mutate',[
    lambda r:r['rows'][0].update(session='2024-01-02'),
    lambda r:r['rows'][0].update(price=999),
    lambda r:r['rows'][0].update(previous_equity=999),
    lambda r:r['metrics'].update(net_return=1.),
    lambda r:r['baselines']['buy_hold']['metrics'].update(net_return=1.),
    lambda r:r['directional'].update(short_net_pnl=1.),
    lambda r:r['rows'].pop(),
])
def test_rejects_mislabeled_future_or_tampered_paths(mutate):
    results,data,config=fixture();mutate(results['SPY-validation'])
    with pytest.raises(ValueError):validate_development_report(results,data,config)

from dataclasses import replace
from datetime import date
import pytest
from rl_stocks.core.contracts import Request
from rl_stocks.data.providers.synthetic.adapter import SyntheticProvider
from rl_stocks.data.validation.overlap import compare_overlap


@pytest.fixture
def bars():return SyntheticProvider().bars(Request(('SPY',),date(2009,12,1),date(2009,12,5)))


def test_retrieval_revision_change_does_not_change_execution_units(bars):
    candidate=[replace(b,retrieved_time='2026-10-20T00:00:00+00:00',revision='later') for b in bars]
    r=compare_overlap(bars,candidate)
    assert r['passed'] and r['rows']==4 and not r['rescaling_applied']


@pytest.mark.parametrize('field',['open','high','low','close','volume','dividend','split','available_time'])
def test_each_execution_field_mismatch_fails(bars,field):
    candidate=list(bars);b=bars[0]
    if field in ('open','high','low','close'):
        candidate[0]=replace(b,open=b.open*2,high=b.high*2,low=b.low*2,close=b.close*2)
    else:candidate[0]=replace(b,**{field:'2009-12-01T21:16:00+00:00' if field=='available_time' else getattr(b,field)+1})
    r=compare_overlap(bars,candidate)
    assert not r['passed'] and r['mismatch_count']>0
    assert not r['reference_modified'] and not r['candidate_modified']


def test_missing_panel_and_invalid_tolerance_rejected(bars):
    with pytest.raises(ValueError):compare_overlap(bars,bars[:-1])
    with pytest.raises(ValueError):compare_overlap(bars,bars,price_atol=float('nan'))


def test_small_fixed_float_tolerance_and_bounded_evidence(bars):
    candidate=[replace(b,dividend=1e-9) for b in bars]
    assert compare_overlap(bars,candidate)['passed']
    r=compare_overlap(bars,[replace(b,open=b.open*2,high=b.high*2,low=b.low*2,close=b.close*2,dividend=1,volume=b.volume+1) for b in bars])
    assert r['mismatch_count']==24 and len(r['mismatches_first20'])==20

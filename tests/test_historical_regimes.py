from dataclasses import replace
from datetime import date
import pytest
from rl_stocks.core.contracts import Request
from rl_stocks.data.providers.synthetic.adapter import SyntheticProvider
from rl_stocks.data.validation.historical import audit_history


@pytest.fixture
def panel():
    bars=[replace(b,provider='yahoo',feed='yahoo',adjustments='split_adjusted_with_dividends',close=100.,open=100.,high=101.,low=99.) for b in SyntheticProvider().bars(Request(('SPY',),date(2001,1,1),date(2002,2,1)))]
    raw=[dict(symbol='SPY',rows=[dict(date=b.session,Open=b.open,High=b.high,Low=b.low,Close=b.close,Volume=b.volume,Dividends=b.dividend,**{'Stock Splits':0.}) for b in bars])]
    return bars,raw,('SPY',),[b.session for b in bars]


def test_complete_panel_retains_raw_split_without_reapplying(panel):
    bars,raw,symbols,days=panel;raw[0]['rows'][100]['Stock Splits']=2.
    r=audit_history(*panel)
    assert r['coverage_complete'] and not r['automatic_merge']
    assert not r['independent_action_verification'] and not r['training_performed']
    assert r['records'][0]['source_splits'][0]['normalized_factor']==1
    years=r['records'][0]['regime_years']
    assert sum(v['ready252'] for v in years.values())==len(bars)-252
    assert sum(v['joint_negative'] for v in years.values())==0


def test_trailing_regime_counts_depend_only_on_past_closes(panel):
    bars,raw,symbols,days=panel
    for i in range(len(bars)):
        value=100-i*.1;bars[i]=replace(bars[i],open=value,high=value+1,low=value-1,close=value)
        raw[0]['rows'][i].update(Open=value,High=value+1,Low=value-1,Close=value)
    years=audit_history(*panel)['records'][0]['regime_years']
    assert sum(v['negative63'] for v in years.values())==len(bars)-63
    assert sum(v['joint_negative'] for v in years.values())==len(bars)-252


@pytest.mark.parametrize('field,value',[('Dividends',-1),('Stock Splits',float('nan')),('Stock Splits',-2),('Capital Gains',1),('Close',102)])
def test_invalid_or_mismatched_actions_and_fields_rejected(panel,field,value):
    panel[1][0]['rows'][0][field]=value
    with pytest.raises(ValueError):audit_history(*panel)


def test_missing_extra_or_duplicate_sources_rejected(panel):
    bars,raw,symbols,days=panel
    with pytest.raises(ValueError):audit_history(bars[1:],raw,symbols,days)
    with pytest.raises(ValueError):audit_history(bars,raw,symbols,days[:-1])
    raw[0]['rows'].append(raw[0]['rows'][0])
    with pytest.raises(ValueError):audit_history(*panel)

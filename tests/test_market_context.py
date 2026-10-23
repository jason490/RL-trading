from dataclasses import replace
import math
import pytest
from test_core import bar
from rl_stocks.features.market.context import MarketContext


def series(symbol,rate=.01):
    from datetime import date,timedelta
    result=[]
    for i in range(27):
        price=100*math.exp(rate*i)
        result.append(replace(bar((date(2020,1,1)+timedelta(days=i)).isoformat(),op=price,high=price,low=price,close=price),symbol=symbol))
    return result


def context(bars):return MarketContext(bars,{s:[] for s in {b.symbol for b in bars}})


def test_exact_price_momentum_relative_volatility_and_warmup():
    spy=series('SPY');own=series('AAPL',.02);m=context(spy+own)
    assert m.at('AAPL',own[19].session,own[19].available_time) is None
    value=m.at('AAPL',own[20].session,own[20].available_time)
    assert value==pytest.approx([.2,0.,.2],abs=1e-14)
    assert m.at('SPY',spy[20].session,spy[20].available_time)[2]==0.


def test_every_prefix_and_changed_future_suffix_preserve_past_values():
    spy=series('SPY');own=series('AAPL',.02);full=context(spy+own)
    for n in range(21,28):
        prefix=context(spy[:n]+own[:n])
        assert prefix.at('AAPL',own[n-1].session,own[n-1].available_time)==full.at('AAPL',own[n-1].session,own[n-1].available_time)
    altered=spy[:21]+[replace(b,open=b.open*3,high=b.high*3,low=b.low*3,close=b.close*3,split=2.) for b in spy[21:]]+own
    assert context(altered).at('AAPL',own[20].session,own[20].available_time)==full.at('AAPL',own[20].session,own[20].available_time)


def test_delay_alignment_and_explicit_gap_reject_without_stale_fill():
    spy=series('SPY');own=series('AAPL');session=own[20].session;time=own[20].available_time
    delayed=spy[:20]+[replace(spy[20],available_time='2020-02-01T22:00:00+00:00')]+spy[21:]
    assert context(delayed+own).at('AAPL',session,time) is None
    assert context(spy[:10]+spy[11:]+own).at('AAPL',session,time) is None
    assert MarketContext(spy+own,{'SPY':[spy[10].session],'AAPL':[]}).at('AAPL',session,time) is None
    with pytest.raises(ValueError):MarketContext(spy+own,{'SPY':[]})
    with pytest.raises(ValueError):MarketContext(own,{'AAPL':[]})


def test_forward_split_continuity_and_dividend_exclusion():
    spy=series('SPY');own=series('AAPL');m=context(spy+own)
    split=own[:10]+[replace(b,open=b.open/2,high=b.high/2,low=b.low/2,close=b.close/2,split=2. if i==10 else 1.,dividend=50.) for i,b in enumerate(own) if i>=10]
    assert context(spy+split).at('AAPL',own[20].session,own[20].available_time)==pytest.approx(m.at('AAPL',own[20].session,own[20].available_time),abs=1e-14)

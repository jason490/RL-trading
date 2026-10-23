"""Fixed20-session SPY price context with explicit session/availability barriers."""
from collections import defaultdict
import math
from statistics import pstdev
from rl_stocks.data.validation.bars import validate,timestamp

NAMES=('market_return_20','market_volatility_20','relative_market_return_20')


class MarketContext:
    def __init__(self,bars,known_missing,benchmark='SPY'):
        validate(bars);grouped=defaultdict(list)
        for b in bars:grouped[b.symbol].append(b)
        if benchmark not in grouped or set(grouped)-set(known_missing):
            raise ValueError('benchmark and explicit missing-session metadata required for every series')
        self.benchmark=benchmark;self.series={};self.windows={}
        for symbol,series in grouped.items():
            self.series[symbol]=series
            factor=1.;prices=[]
            for b in series:
                factor*=b.split;value=b.close*factor
                if not math.isfinite(value) or value<=0:raise ValueError('finite forward split-continuous prices required')
                prices.append(value)
            logs=[math.log(v) for v in prices]
            available=[timestamp(b.available_time) for b in series];windows={}
            for i in range(20,len(series)):
                first=series[i-20].session;last=series[i].session
                if any(first<day<=last for day in known_missing[symbol]):continue
                momentum=logs[i]-logs[i-20]
                volatility=pstdev([logs[j]-logs[j-1] for j in range(i-19,i+1)]) if symbol==benchmark else None
                windows[last]=(tuple(b.session for b in series[i-20:i+1]),max(available[i-20:i+1]),momentum,volatility)
            self.windows[symbol]=windows

    def at(self,symbol,session,decision_time):
        if symbol not in self.series:raise ValueError('unknown context symbol')
        own=self.windows[symbol].get(session);market=self.windows[self.benchmark].get(session)
        if own is None or market is None:return None
        decision=timestamp(decision_time)
        if own[0]!=market[0] or max(own[1],market[1])>decision:return None
        return [market[2],market[3],own[2]-market[2]]

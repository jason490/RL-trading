"""Same-session sector-ETF momentum with availability and missing-period masks."""
from dataclasses import replace
import math
from rl_stocks.data.universe.config import SECTORS
from rl_stocks.data.universe.sectors import sector_asof


class SectorRotation:
    def __init__(self, bars, proxies, benchmark="SPY", records=(), missing=None,
                 use_classifications=True):
        if set(proxies) != set(SECTORS) or len(set(proxies.values())) != len(SECTORS):
            raise ValueError("one distinct proxy required for each of 11 sectors")
        self.proxies, self.benchmark, self.records = proxies, benchmark, records
        self.use_classifications = use_classifications
        self.series, self.indices = {}, {}
        missing = missing or {}
        for symbol in {b.symbol for b in bars} | {benchmark, *proxies.values()}:
            source = [b for b in bars if b.symbol == symbol]
            if not source:
                raise ValueError(f"missing rotation context {symbol}")
            factor, normalized = 1., []
            for b in source:
                factor *= b.split
                normalized.append(replace(b, close=b.close*factor, dividend=b.dividend*factor))
            self.series[symbol] = normalized
            self.indices[symbol] = {b.session: i for i, b in enumerate(normalized)}
        self.missing = missing
        self.names = [f"sector_{sector}_{field}" for sector in SECTORS for field in
                      ("ready", "return_20", "relative_20", "relative_60", "rank_20")]
        if use_classifications:
            self.names += ["stock_sector_known"]+[f"stock_sector_{s}" for s in SECTORS]
            self.names += ["stock_sector_relative_20"]

    def momentum(self, symbol, session, decision_time, window):
        index = self.indices[symbol].get(session)
        if index is None or index < window:
            return None
        series = self.series[symbol][index-window:index+1]
        benchmark_index = self.indices[self.benchmark].get(session)
        if benchmark_index is None or benchmark_index < window:
            return None
        benchmark_sessions = [b.session for b in self.series[self.benchmark][
            benchmark_index-window:benchmark_index+1]]
        if [b.session for b in series] != benchmark_sessions:
            return None
        if any(b.available_time > decision_time for b in series) or any(
                series[0].session < day <= session for day in self.missing.get(symbol, [])):
            return None
        # Cash-dividend total return, without reconstructing future adjusted close.
        return sum(math.log((b.close+b.dividend)/a.close) for a, b in zip(series, series[1:]))

    def at(self, symbol, session, decision_time):
        benchmark = [self.momentum(self.benchmark, session, decision_time, n) for n in (20, 60)]
        signals = {}
        for sector, proxy in self.proxies.items():
            values = [self.momentum(proxy, session, decision_time, n) for n in (20, 60)]
            if None not in benchmark+values:
                signals[sector] = (values[0], values[0]-benchmark[0], values[1]-benchmark[1])
        result = []
        for sector in SECTORS:
            if sector not in signals:
                result.extend([0.]*5)
                continue
            value = signals[sector]
            # Rank uses only currently ready proxies; equal values tie at midpoint.
            less = sum(v[1] < value[1] for v in signals.values())
            equal = sum(v[1] == value[1] for v in signals.values())
            rank = (less+(equal-1)/2)/max(1, len(signals)-1)
            result.extend([1., *value, rank])
        if not self.use_classifications:
            return result
        sector = sector_asof(self.records, symbol, session, decision_time)
        stock_return_20 = self.momentum(symbol, session, decision_time, 20)
        known = sector in signals and stock_return_20 is not None
        result += [float(known)]+[float(known and s == sector) for s in SECTORS]
        result += [stock_return_20-signals[sector][0] if known and stock_return_20 is not None else 0.]
        return result

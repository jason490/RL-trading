"""Opt-in aligned stock/market/sector relationships; never enable model inputs."""

import math
from statistics import mean

from rl_stocks.data.validation.bars import timestamp
from rl_stocks.features.market.context import MarketContext

NAMES = ("stock_market_correlation_20", "stock_sector_correlation_20",
         "sector_return_20", "relative_sector_return_20")


def correlation(left, right):
    a, b = mean(left), mean(right)
    numerator = sum((x - a) * (y - b) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(sum((x - a) ** 2 for x in left)
                            * sum((y - b) ** 2 for y in right))
    return None if denominator == 0 else max(-1.0, min(1.0, numerator / denominator))


class CrossStockRelationships:
    """Caller owns historical sector mapping and invokes separately per split.

    Only exact common21-session windows are usable. Missing or unpublished
    observations return None; zero-variance correlation is missing, not zero.
    """

    def __init__(self, bars, known_missing, sector_proxies, benchmark="SPY"):
        self.context = MarketContext(bars, known_missing, benchmark)
        self.proxies = dict(sector_proxies)
        if any(s not in self.context.series or p not in self.context.series
               for s, p in self.proxies.items()):
            raise ValueError("stock and sector proxy histories required")
        self.returns = {}
        for symbol, series in self.context.series.items():
            for i in range(20, len(series)):
                self.returns[symbol, series[i].session] = tuple(
                    math.log(series[j].close * series[j].split / series[j - 1].close)
                    for j in range(i - 19, i + 1))

    def at(self, symbol, session, decision_time):
        if symbol not in self.proxies:
            raise ValueError("explicit historical sector proxy mapping required")
        symbols = (symbol, self.context.benchmark, self.proxies[symbol])
        windows = [self.context.windows[s].get(session) for s in symbols]
        if any(w is None for w in windows):
            return None
        if any(w[0] != windows[0][0] for w in windows):
            return None
        if max(w[1] for w in windows) > timestamp(decision_time):
            return None
        own, market, sector = [self.returns[s, session] for s in symbols]
        a, b = correlation(own, market), correlation(own, sector)
        if a is None or b is None:
            return None
        return [a, b, sum(sector), sum(own) - sum(sector)]

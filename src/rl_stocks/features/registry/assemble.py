import math
from dataclasses import replace
from rl_stocks.features.price.indicators import returns, volatility, atr
from rl_stocks.features.trend.ema import ema
from rl_stocks.features.momentum.rsi import rsi
from rl_stocks.features.volume.relative import relative_volume
from rl_stocks.features.timeframes.weekly import completed_weekly

FEATURE_SCHEMA = "daily-indicators-v1"
NAMES = ("return_1", "volatility_20", "ema_20_ratio", "ema_50_ratio", "rsi_14", "atr_14_ratio", "relative_volume_20")

def assemble(bars, weekly=False):
    factors=[]
    factor=1.
    for b in bars:
        factor*=b.split
        factors.append(factor)
    # Forward-only split continuity: today's effective split is known today.
    # ATR is converted back to executable price units below.
    bars=[replace(b,open=b.open*f,high=b.high*f,low=b.low*f,close=b.close*f,
                  volume=b.volume/f) for b,f in zip(bars,factors)]
    prices = [b.close for b in bars]
    ret = returns(prices)
    vol = volatility(ret)
    e20, e50 = ema(prices,20), ema(prices,50)
    rs = rsi(prices)
    at = atr(bars)
    rv = relative_volume([b.volume for b in bars])
    week = completed_weekly(bars) if weekly else None
    result = []
    for i,b in enumerate(bars):
        values = [ret[i],vol[i], prices[i]/e20[i]-1 if e20[i] else None,
                  prices[i]/e50[i]-1 if e50[i] else None,
                  rs[i]/100 if rs[i] is not None else None,
                  at[i]/prices[i] if at[i] else None, rv[i]]
        if weekly:
            values.append(week[i])
        if all(v is not None and math.isfinite(v) for v in values):
            result.append({"bar": i, "values": values, "atr": at[i]/factors[i],
                "available_time": b.available_time})
    return result, list(NAMES)+(["completed_weekly_ema10_ratio"] if weekly else [])

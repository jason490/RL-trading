import math
from dataclasses import replace
from rl_stocks.features.price.indicators import returns, volatility, atr
from rl_stocks.features.trend.ema import ema
from rl_stocks.features.momentum.rsi import rsi
from rl_stocks.features.volume.relative import relative_volume
from rl_stocks.features.timeframes.weekly import completed_weekly

FEATURE_SCHEMA = "daily-indicators-v1"
NAMES = ("return_1", "volatility_20", "ema_20_ratio", "ema_50_ratio", "rsi_14", "atr_14_ratio", "relative_volume_20")

def assemble(bars, weekly=False, zone_timeframes=(), zone_config=None, trend_context="none", session_mode="none"):
    from rl_stocks.features.price.sessions import MODES, NAMES as SESSION_NAMES, session_context
    if session_mode not in MODES:raise ValueError('unknown session context mode')
    from rl_stocks.features.trend.context import validate_mode, trailing_context, NAMES as CONTEXT_NAMES
    validate_mode(trend_context)
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
    sessions=session_context(bars) if session_mode!='none' else None
    context = trailing_context(prices) if trend_context != "none" else None
    ret = returns(prices)
    vol = volatility(ret)
    e20, e50 = ema(prices,20), ema(prices,50)
    rs = rsi(prices)
    at = atr(bars)
    rv = relative_volume([b.volume for b in bars])
    week = completed_weekly(bars) if weekly else None
    if zone_timeframes:
        from rl_stocks.features.market_structure.supply_demand import supply_demand, ZoneConfig
        zones, zone_events = supply_demand(bars, zone_timeframes, zone_config or ZoneConfig())
        events_by_time = {}
        for timeframe, events in zone_events.items():
            for event in events:
                events_by_time.setdefault(event["available_time"], []).append(
                    {"timeframe": timeframe, **event})
    result = []
    for i,b in enumerate(bars):
        if context is not None and context[i] is None:
            continue
        if sessions is not None and sessions[i] is None:continue
        values = [ret[i],vol[i], prices[i]/e20[i]-1 if e20[i] else None,
                  prices[i]/e50[i]-1 if e50[i] else None,
                  rs[i]/100 if rs[i] is not None else None,
                  at[i]/prices[i] if at[i] else None, rv[i]]
        if trend_context in ("masked", "full"):
            values.extend(context[i] if trend_context == "full" else [0., 0., 0.])
        if sessions is not None:values.extend(sessions[i] if session_mode=='full' else [0.]*4)
        if weekly:
            values.append(week[i])
        if zone_timeframes:
            values.extend(zones[i]["values"])
        if all(v is not None and math.isfinite(v) for v in values):
            result.append({"bar": i, "values": values, "atr": at[i]/factors[i],
                "available_time": b.available_time})
            if zone_timeframes:
                result[-1]["zones"] = zones[i]["zones"]
                result[-1]["zone_events"] = events_by_time.get(b.available_time, [])
    names = list(NAMES)+(list(CONTEXT_NAMES) if trend_context in ("masked", "full") else [])+(list(SESSION_NAMES) if sessions is not None else [])+(["completed_weekly_ema10_ratio"] if weekly else [])
    if zone_timeframes:
        from rl_stocks.features.market_structure.supply_demand import feature_names
        names.extend(feature_names(zone_timeframes))
    return result, names

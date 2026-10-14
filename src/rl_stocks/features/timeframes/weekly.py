from datetime import date
from rl_stocks.features.trend.ema import ema

def completed_weekly(bars):
    # A week is released at the first observed session of the following week.
    # This conservative rule works with holidays and never publishes a partial week.
    groups, result, last_key = [], [], None
    current = []
    value = None
    for b in bars:
        d = date.fromisoformat(b.session)
        key = d.isocalendar()[:2]
        if last_key is not None and key != last_key:
            groups.append(current[-1].close)
            if len(groups)>=10:
                value = groups[-1]/ema(groups,10)[-1]-1
            current = []
        current.append(b)
        result.append(value)
        last_key = key
    return result

"""Forward-only completed exchange-week OHLCV release at daily decisions."""
from dataclasses import replace
from datetime import date


def completed_week_bars(bars):
    current, last_key, periods = [], None, 0
    for bar in bars:
        key = date.fromisoformat(bar.session).isocalendar()[:2]
        completed = None
        if last_key is not None and key != last_key:
            # History/gap boundaries may begin midweek; discard that first group.
            if periods:
                completed = replace(current[-1], open=current[0].open,
                    high=max(b.high for b in current), low=min(b.low for b in current),
                    volume=sum(b.volume for b in current), split=1., dividend=0.,
                    available_time=max(b.available_time for b in current))
            periods += 1
            current = []
        current.append(bar)
        last_key = key
        yield completed

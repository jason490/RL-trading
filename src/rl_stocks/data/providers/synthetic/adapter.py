from datetime import datetime, timedelta, timezone
import math
import random
from rl_stocks.core.contracts import Bar, Capabilities, Request

class SyntheticProvider:
    capabilities = Capabilities("synthetic", feed="synthetic", limitations="Weekdays only; not an exchange calendar; no market evidence")
    def __init__(self, seed=11):
        self.seed = seed
    def bars(self, request: Request):
        result = []
        for n, symbol in enumerate(request.symbols):
            rng = random.Random(self.seed + n)
            price = 100.0 + n * 30
            day = request.start
            while day < request.end:
                if day.weekday() < 5:
                    op = price * math.exp(rng.gauss(0, .004))
                    close = op * math.exp(rng.gauss(.0002, .012))
                    low = min(op, close) * (1-rng.uniform(.001, .012))
                    high = max(op, close) * (1+rng.uniform(.001, .012))
                    base = datetime.combine(day, datetime.min.time(), timezone.utc)
                    o = base + timedelta(hours=14, minutes=30)
                    e = base + timedelta(hours=21)
                    result.append(Bar(symbol, str(day), o.isoformat(), e.isoformat(),
                        (e+timedelta(minutes=15)).isoformat(), "2026-10-14T00:00:00+00:00",
                        "synthetic", "synthetic", f"seed-{self.seed}", op, high, low, close,
                        rng.randint(100000, 5000000)))
                    price = close
                day += timedelta(days=1)
        return result

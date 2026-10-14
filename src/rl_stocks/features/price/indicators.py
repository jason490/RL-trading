import math
from statistics import pstdev

def returns(prices):
    return [None]+[math.log(b/a) for a,b in zip(prices, prices[1:])]

def volatility(values, window=20):
    return [pstdev(values[i-window+1:i+1]) if i>=window and
            all(v is not None for v in values[i-window+1:i+1]) else None
            for i in range(len(values))]

def atr(bars, window=14):
    ranges = [max(b.high-b.low, abs(b.high-bars[i-1].close), abs(b.low-bars[i-1].close))
              if i else b.high-b.low for i,b in enumerate(bars)]
    result = [None]*len(bars)
    value = None
    for i, tr in enumerate(ranges):
        if i == window-1:
            value = sum(ranges[:window])/window
        elif i >= window:
            value = (value*(window-1)+tr)/window
        if value is not None:
            result[i] = value
    return result

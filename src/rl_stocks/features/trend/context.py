"""Trailing price context on the caller's forward-only split-continuous prices."""
import math
from collections import deque

MODES = ('none', 'matched_baseline', 'masked', 'full')
NAMES = ('price_log_return_63', 'price_log_return_252', 'price_drawdown_252')


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError('unknown trend context mode')
    return mode


def trailing_context(prices):
    """Require 253 contiguous observations; high includes today's close.

    Price returns exclude cash dividends. No future prices, future decline
    labels, cross-symbol ranks or fitted thresholds enter these values.
    """
    high = deque()
    result = []
    for i, price in enumerate(prices):
        if not math.isfinite(price) or price <= 0:
            raise ValueError('positive finite context prices required')
        while high and high[0] <= i - 252:
            high.popleft()
        while high and prices[high[-1]] <= price:
            high.pop()
        high.append(i)
        result.append(None if i < 252 else [math.log(price / prices[i-63]),
            math.log(price / prices[i-252]), price / prices[high[0]] - 1.])
    return result

def ema(prices, window):
    result, value = [], prices[0]
    alpha = 2/(window+1)
    for i,p in enumerate(prices):
        value = alpha*p+(1-alpha)*value
        result.append(value if i >= window-1 else None)
    return result

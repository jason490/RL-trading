def rsi(prices, window=14):
    result = [None]*len(prices)
    gains, losses = [], []
    up = down = None
    for i in range(1,len(prices)):
        d = prices[i]-prices[i-1]
        gains.append(max(d,0)); losses.append(max(-d,0))
        if i == window:
            up, down = sum(gains)/window, sum(losses)/window
        elif i > window:
            up = (up*(window-1)+gains[-1])/window
            down = (down*(window-1)+losses[-1])/window
        if up is not None:
            result[i] = 50.0 if up==down==0 else (100.0 if down==0 else 100-100/(1+up/down))
    return result

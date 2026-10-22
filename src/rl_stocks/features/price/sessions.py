"""Completed-session price decomposition on forward split-continuous bars."""
import math

MODES=('none','masked','full')
NAMES=('overnight_log_return','intraday_log_return','overnight_ema_20','intraday_ema_20')


def session_context(bars):
    """Require20 completed pairs; caller resets chunks across known gaps.

    Price-only, excludes dividends. Input OHLC already has forward-only split
    continuity. All four fields are released at the current bar availability.
    """
    result=[];gap_mean=day_mean=None;alpha=2/21
    for i,b in enumerate(bars):
        if any(not math.isfinite(v) or v<=0 for v in [b.open,b.close]):
            raise ValueError('positive finite session prices required')
        if i==0:result.append(None);continue
        gap=math.log(b.open/bars[i-1].close);day=math.log(b.close/b.open)
        gap_mean=gap if gap_mean is None else alpha*gap+(1-alpha)*gap_mean
        day_mean=day if day_mean is None else alpha*day+(1-alpha)*day_mean
        result.append(None if i<20 else [gap,day,gap_mean,day_mean])
    return result

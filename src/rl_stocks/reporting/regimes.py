"""Training-only, trailing price-regime coverage; never labels future recoveries."""
from collections import Counter


def regime_coverage(dataset,universe,long_window=252,short_window=63,decline_threshold=-.2):
    if not 0<short_window<long_window or not -1<decline_threshold<0:raise ValueError('invalid regime settings')
    records=[]
    for episode in dataset['episodes'].values():
        if episode['split']!='train':continue
        bars=episode['bars']
        if any(b['session']>'2020-12-31' for b in bars):raise ValueError('training split crosses frozen boundary')
        if any(b['adjustments']!='split_adjusted_with_dividends' or b['split']!=1 for b in bars):
            raise ValueError('audit requires verified split-adjusted price inputs')
        counts=Counter();run=longest=0;first=last=None;worst=None
        for i in range(long_window,len(bars)):
            trailing=bars[i]['close']/bars[i-long_window]['close']-1
            recent=bars[i]['close']/bars[i-short_window]['close']-1
            bucket='declining' if trailing<decline_threshold else 'rising' if trailing>-decline_threshold else 'middle'
            counts[bucket]+=1;worst=trailing if worst is None else min(worst,trailing)
            persistent=trailing<decline_threshold and recent<0
            run=run+1 if persistent else 0;longest=max(longest,run)
            if persistent:
                counts['declining_both_windows']+=1
                first=first or bars[i]['session'];last=bars[i]['session']
        n=sum(counts[k] for k in ('declining','rising','middle'))
        records.append({'symbol':episode['symbol'],'training_first':bars[0]['session'],'training_last':bars[-1]['session'],
            'eligible_sessions':n,'declining_sessions':counts['declining'],'rising_sessions':counts['rising'],
            'middle_sessions':counts['middle'],'declining_both_windows_sessions':counts['declining_both_windows'],
            'longest_consecutive_declining_both_windows':longest,'worst_trailing_price_return':worst,
            'first_observed_declining_both_windows':first,'last_observed_declining_both_windows':last})
    if not records:raise ValueError('training episodes required')
    totals={key:sum(r[key] for r in records) for key in ('eligible_sessions','declining_sessions','rising_sessions',
        'middle_sessions','declining_both_windows_sessions')}
    return {'split':'train','long_window':long_window,'short_window':short_window,'decline_threshold':decline_threshold,
        'records':sorted(records,key=lambda r:r['symbol']),'totals':totals,
        'symbols_with_declining_both_windows':sum(r['declining_both_windows_sessions']>0 for r in records),
        'universe_survivorship_statement':universe['survivorship'],'selected_on':universe['selected_on'],
        'historical_membership_verified':False,'delisting_returns_verified':False,
        'interpretation':'Trailing split-adjusted price trends only; dividends excluded. Training-only coverage diagnostics, not a future recovery label, total-return backtest or bias-free universe.'}

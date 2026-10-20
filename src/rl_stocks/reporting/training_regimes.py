"""Reporting-only trailing-decline/forward-outcome accounting on fitted paths."""
import math
from statistics import mean


def summarize_training_regimes(results, horizon=20, threshold=.1):
    if horizon < 1 or not 0 < threshold < 1:
        raise ValueError('positive horizon and fractional threshold required')
    groups = {k: [] for k in ('all', 'past_decline', 'past_decline_future_decline',
                              'past_decline_future_rebound', 'past_decline_future_other')}
    for result in results.values():
        rows = result['rows']
        if result['split'] != 'train' or not rows:
            raise ValueError('training paths only')
        dates = [r['session'] for r in rows]
        if dates != sorted(set(dates)) or any(not '2018-01-01' <= d <= '2020-12-02' for d in dates):
            raise ValueError('unique fixed fitted-period sessions required')
        prices = [rows[0]['decision']['reference']] + [r['price'] for r in rows[:-1]]
        if any(not math.isfinite(v) or v <= 0 for v in prices):
            raise ValueError('positive finite decision prices required')
        for i in range(horizon, len(rows)-horizon):
            row = rows[i]
            if row['decision'] is None:
                continue  # Rejected protection has no approved target decision.
            if not math.isclose(row['decision']['reference'],prices[i],rel_tol=1e-9):
                raise ValueError('decision reference must match prior close')
            if row['decision']['decision_time'][:10] >= row['session']:
                raise ValueError('prior-session decision required')
            exposure = row['decision']['approved'][0]
            if not math.isfinite(exposure) or abs(exposure) > 1:
                raise ValueError('bounded approved exposure required')
            past = prices[i]/prices[i-horizon]-1
            future = prices[i+horizon]/prices[i]-1
            entry = dict(exposure=exposure, future=future, past=past)
            groups['all'].append(entry)
            if past <= -threshold:
                groups['past_decline'].append(entry)
                suffix = 'decline' if future <= -threshold else 'rebound' if future >= threshold else 'other'
                groups['past_decline_future_'+suffix].append(entry)
    return {name: dict(count=len(items),
        mean_requested_exposure=mean(x['exposure'] for x in items) if items else None,
        short_request_fraction=mean(x['exposure'] < 0 for x in items) if items else None,
        mean_future_price_return=mean(x['future'] for x in items) if items else None)
        for name, items in groups.items()}

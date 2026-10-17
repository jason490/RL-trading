"""Bitemporal event lookup: effective date AND public availability are required."""
from functools import lru_cache
from rl_stocks.data.validation.bars import timestamp


@lru_cache(maxsize=16384)
def at(value):
    return timestamp(value)


def asof_versions(records, decision_time):
    decision = at(decision_time)
    latest = {}
    for row in records:
        if at(row['available_time']) <= decision:
            key = row['event_id']
            if key not in latest or at(latest[key]['available_time']) < at(row['available_time']):
                latest[key] = row
    return list(latest.values())


def state_asof(records, security_id, session, decision_time, field, default=None):
    candidates = [r for r in asof_versions(records, decision_time)
                  if r['security_id'] == security_id and r['effective_from'] <= session]
    if not candidates:
        return default
    effective = max(r['effective_from'] for r in candidates)
    values = {r[field] for r in candidates if r['effective_from'] == effective}
    if len(values) != 1:
        raise ValueError('conflicting same-effective-date events')
    return values.pop()


def member_asof(payload, security_id, session, decision_time):
    return state_asof(payload['membership'], security_id, session, decision_time, 'member', False)


def ticker_asof(payload, security_id, session, decision_time):
    return state_asof(payload['aliases'], security_id, session, decision_time, 'ticker')


def resolve_ticker(payload, ticker, session, decision_time):
    matches = [r['security_id'] for r in payload['securities']
               if ticker_asof(payload, r['security_id'], session, decision_time) == ticker
               and r['listed_from'] <= session
               and (r['last_trade_session'] is None or session <= r['last_trade_session'])]
    if len(matches) != 1:
        raise ValueError('ticker must resolve to one security at this date')
    return matches[0]

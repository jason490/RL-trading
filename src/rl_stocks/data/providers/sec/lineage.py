"""Causal issuer-evidence queries; no implicit historical price or financial joins."""
import re
from datetime import date, datetime, timedelta, timezone

import exchange_calendars as xc

AVAILABILITY_POLICY = 'sec-source-dates-plus-two-xnys-sessions-v1'


def _day(value):
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError('exact ISO session/date required')
    return result


def _time(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('aware availability/decision timestamp required')
    return result.astimezone(timezone.utc)


def conservative_availability(filed, accepted_source):
    """Reconstruct a daily-use lag; do not claim exact first public availability.

    SEC source-local *dates* avoid guessing whether a naive acceptance hour uses
    standard/daylight time. Wait until the second XNYS open strictly after both
    original filing date and acceptance date. The variable SEC dissemination lag
    has no guaranteed bound: this is an explicit research assumption.
    """
    filing = _day(filed)
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?', accepted_source):
        raise ValueError('original naive SEC acceptance timestamp required')
    accepted = datetime.fromisoformat(accepted_source).date()
    latest = max(filing, accepted)
    if latest >= date(2024, 1, 1):
        raise ValueError('retired source filing/acceptance rejected')
    limit = latest + timedelta(days=30)
    calendar = xc.get_calendar('XNYS', start=latest.isoformat(), end=limit.isoformat())
    sessions = [s for s in calendar.sessions if s.date() > latest]
    return dict(available_time=calendar.session_open(sessions[1]).isoformat().replace('+00:00', 'Z'),
                policy=AVAILABILITY_POLICY, exact_first_publication_verified=False,
                availability_reconstructed=True)


def issuer_asof(records, symbol, session, decision_time):
    """Return visible document/event evidence, never backfill an unknown issuer.

    An observation anchors its own effective session, not an earlier reporting
    period. Changing CIK requires an explicit successor link; a later subsidiary
    filing must not silently replace the known public parent. Source/stock-price
    linkage and financial comparability remain separate, unverified requirements.
    """
    query_day, decision = _day(session), _time(decision_time)
    visible = []
    for r in records:
        if r['symbol'] != symbol:
            continue
        effective, available = _day(r['effective_session']), _time(r['available_time'])
        if effective > query_day or available > decision:
            continue
        if r['kind'] not in {'observation', 'successor'}:
            raise ValueError('explicit issuer evidence kind required')
        if not re.fullmatch(r'[1-9][0-9]{0,9}', r['cik']):
            raise ValueError('valid normalized issuer CIK required')
        if (not re.fullmatch(r'[a-f0-9]{64}', r['source_sha256'])
                or not r['source_url'].startswith('https://www.sec.gov/Archives/')):
            raise ValueError('original source checksum and SEC filing URL required')
        if r['kind'] == 'successor' and (not re.fullmatch(r'[1-9][0-9]{0,9}', r['predecessor_cik'])
                                         or r['predecessor_cik'] == r['cik']):
            raise ValueError('distinct explicit predecessor CIK required')
        visible.append(r)
    visible.sort(key=lambda r: (r['effective_session'], _time(r['available_time']),
                                r['kind'] == 'successor'))
    current = None
    for r in visible:
        if current is not None and r['cik'] != current['cik']:
            if r['kind'] != 'successor' or r['predecessor_cik'] != current['cik']:
                raise ValueError('unresolved/conflicting issuer transition')
        current = r
    if current is None:
        return None
    return dict(cik=current['cik'], source_sha256=current['source_sha256'],
                effective_session=current['effective_session'], available_time=current['available_time'],
                historical_price_link_verified=False, financial_comparability_verified=False,
                financial_features_enabled=False)

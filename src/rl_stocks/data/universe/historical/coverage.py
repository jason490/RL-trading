"""Coverage cannot be inferred from successful downloads or surviving symbols."""
from collections import Counter, defaultdict
from datetime import timedelta
import math
from rl_stocks.core.contracts import digest
from rl_stocks.data.validation.bars import timestamp
from rl_stocks.data.universe.historical.schema import validate, ATTESTATIONS, finite
from rl_stocks.data.universe.historical.registry import state_asof, asof_versions


def terminal_proceeds(event, last_close):
    """Extra settlement value after the last marked close, never a feature."""
    if not event['verified'] or event['kind'] not in ('cash_per_share', 'delisting_return'):
        raise ValueError('unverified/unresolved/non-cash terminal outcome')
    finite(last_close, 0.)
    finite(event['reference_price'], 0.)
    finite(event['value'], 0. if event['kind'] == 'cash_per_share' else -1.)
    if last_close <= 0 or event['basis'] != 'additional_to_last_close':
        raise ValueError('invalid settlement basis/reference')
    if not math.isclose(event['reference_price'], last_close, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError('terminal reference does not match last marked close')
    return event['value'] if event['kind'] == 'cash_per_share' else last_close*(1+event['value'])


def audit(payload):
    validate(payload)
    import exchange_calendars as xc
    universe = payload['universe']
    calendar = xc.get_calendar('XNYS', start=universe['start'], end=universe['end'])
    sessions = [s for s in calendar.sessions if universe['start'] <= str(s.date()) < universe['end']]
    problems = []
    prices = defaultdict(dict)
    events = {table: defaultdict(list) for table in ('aliases', 'membership', 'terminals')}
    securities = {r['security_id']: r for r in payload['securities']}
    for table in events:
        for row in payload[table]:
            events[table][row['security_id']].append(row)
    for row in payload['prices']:
        prices[row['security_id']][row['bar']['session']] = row['bar']
    for name in ATTESTATIONS:
        if not payload['attestations'][name]['verified']:
            problems.append({'code': 'unverified_'+name})
    stats = {sid: {'security_id':sid, 'eligible_sessions':0, 'covered_sessions':0,
                  'missing_sessions':[], 'terminal_required':False,
                  'terminal_covered':False} for sid in securities}
    panel = []
    for session in sessions:
        day = str(session.date())
        close = calendar.session_close(session)
        decision = (close+timedelta(minutes=15)).isoformat()
        occupied = {}
        for sid, security in securities.items():
            try:
                member = state_asof(events['membership'][sid], sid, day, decision, 'member', False)
            except ValueError:
                problems.append({'code':'ambiguous_membership','security_id':sid,'session':day})
                continue
            if not member:
                continue
            entry = stats[sid]
            entry['eligible_sessions'] += 1
            try:
                ticker = state_asof(events['aliases'][sid], sid, day, decision, 'ticker')
            except ValueError:
                ticker = None
                problems.append({'code':'ambiguous_asof_alias','security_id':sid,'session':day})
            if ticker is None:
                problems.append({'code':'missing_asof_alias','security_id':sid,'session':day})
            elif ticker in occupied and occupied[ticker] != sid:
                problems.append({'code':'ambiguous_asof_ticker','security_id':sid,'session':day})
            else:
                occupied[ticker] = sid
            bar = prices[sid].get(day)
            covered = bar is not None
            if day < security['listed_from'] or security['last_trade_session'] and day > security['last_trade_session']:
                problems.append({'code':'membership_outside_listing','security_id':sid,'session':day})
            if covered:
                if bar['symbol'] != ticker:
                    problems.append({'code':'bar_ticker_identity_mismatch','security_id':sid,'session':day})
                if timestamp(bar['event_time']) != timestamp(close.isoformat()) or timestamp(bar['open_time']) != timestamp(calendar.session_open(session).isoformat()):
                    problems.append({'code':'bar_calendar_mismatch','security_id':sid,'session':day})
                if timestamp(bar['available_time']) > timestamp(decision):
                    problems.append({'code':'price_unavailable_at_decision','security_id':sid,'session':day})
                entry['covered_sessions'] += 1
            else:
                entry['missing_sessions'].append(day)
                problems.append({'code':'missing_member_price','security_id':sid,'session':day})
            panel.append({'security_id':sid,'session':day,'decision_time':decision,
                          'ticker':ticker,'price_covered':covered})
    # All cohort securities whose lifecycle ends within the requested interval
    # need an outcome, even if an index removed them earlier. Do not hide failures
    # by trimming the requested prices to surviving membership days.
    for sid, security in securities.items():
        last = security['last_trade_session']
        if stats[sid]['eligible_sessions'] and security['status'] == 'unknown':
            problems.append({'code':'unknown_lifecycle_status','security_id':sid})
        if stats[sid]['eligible_sessions'] and security['status'] == 'delisted' and last is None:
            problems.append({'code':'missing_last_trade_session','security_id':sid})
        if last is None or not universe['start'] <= last < universe['end'] or not stats[sid]['eligible_sessions']:
            continue
        entry = stats[sid]
        entry['terminal_required'] = True
        candidates = events['terminals'][sid]
        # End-of-evidence outcome revisions are audit labels, never historical inputs.
        if not candidates:
            problems.append({'code':'missing_terminal_outcome','security_id':sid})
            continue
        cutoff = max(timestamp(r['retrieved_time']) for r in candidates).isoformat()
        latest = asof_versions(candidates, cutoff)
        if len(latest) != 1:
            problems.append({'code':'ambiguous_terminal_outcome','security_id':sid})
            continue
        event = latest[0]
        bar = prices[sid].get(last)
        if bar is None:
            problems.append({'code':'missing_last_trade_price','security_id':sid})
            continue
        if event['last_trade_session'] != last:
            problems.append({'code':'terminal_date_mismatch','security_id':sid})
            continue
        try:
            terminal_proceeds(event, bar['close'])
        except ValueError as error:
            problems.append({'code':'unusable_terminal_outcome','security_id':sid,'reason':str(error)})
            continue
        if timestamp(event['settlement_time']) < timestamp(bar['event_time']):
            problems.append({'code':'terminal_precedes_last_close','security_id':sid})
            continue
        entry['terminal_covered'] = True
    selected = [r for r in stats.values() if r['eligible_sessions']]
    if not selected:
        problems.append({'code':'empty_eligible_universe'})
    excluded = [r['security_id'] for r in stats.values() if not r['eligible_sessions']]
    totals = {'securities_in_master':len(securities), 'selected_securities':len(selected),
              'unselected_master_securities':len(excluded), 'sessions':len(sessions),
              'expected_member_sessions':sum(r['eligible_sessions'] for r in selected),
              'covered_member_sessions':sum(r['covered_sessions'] for r in selected),
              'terminal_outcomes_required':sum(r['terminal_required'] for r in selected),
              'terminal_outcomes_covered':sum(r['terminal_covered'] for r in selected)}
    return {'version':1,'history_id':digest(payload),'universe':universe,
            'coverage_passed':not problems,'training_ready':False,
            'training_ready_reason':'Historical membership/terminal-aware episode integration is a separate gate; this exports evidence and eligible session requests only.',
            'totals':totals,'issue_counts':dict(Counter(r['code'] for r in problems)),
            'problems':problems,'securities':list(stats.values()),'unselected_security_ids':excluded,
            'eligible_panel':panel,'selection_uses_terminal_outcomes':False,
            'interpretation':'Coverage of the supplied evidence, conditional on source attestations; not independent certification of an entire market universe.'}

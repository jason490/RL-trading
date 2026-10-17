"""Strict file interchange for vendor-neutral historical security evidence."""
from datetime import date
import math
from rl_stocks.core.contracts import Bar
from rl_stocks.data.validation.bars import timestamp, validate as validate_bars

TABLES = ('securities', 'aliases', 'membership', 'terminals', 'prices')
ATTESTATIONS = ('membership_complete', 'identity_verified', 'membership_vintages_verified',
                'corporate_actions_verified', 'delisted_price_coverage_verified')
COMMON = {'event_id', 'security_id', 'source', 'revision', 'available_time', 'retrieved_time'}


def day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('invalid ISO session date')
    return value


def provenance(record):
    for key in ('source', 'revision', 'available_time', 'retrieved_time'):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError('record provenance required: '+key)
    if timestamp(record['available_time']) > timestamp(record['retrieved_time']):
        raise ValueError('retrieval precedes recorded availability')


def finite(value, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('finite numeric outcome required')
    if minimum is not None and value < minimum:
        raise ValueError('outcome below permitted bound')


def validate(payload):
    if set(payload) != {'version', 'universe', 'attestations', *TABLES} or payload['version'] != 1:
        raise ValueError('historical payload schema mismatch')
    universe = payload['universe']
    if set(universe) != {'name', 'start', 'end', 'calendar', 'selection_rule', 'source'}:
        raise ValueError('universe schema mismatch')
    if not all(isinstance(universe[k], str) and universe[k] for k in ('name', 'selection_rule', 'source')):
        raise ValueError('declared universe/source/selection rule required')
    if day(universe['start']) >= day(universe['end']) or universe['calendar'] != 'XNYS':
        raise ValueError('invalid historical range/calendar')
    if set(payload['attestations']) != set(ATTESTATIONS):
        raise ValueError('all evidence attestations must be explicit')
    for value in payload['attestations'].values():
        if set(value) != {'verified', 'source', 'revision'} or type(value['verified']) is not bool:
            raise ValueError('invalid attestation')
        if value['verified'] and not (value['source'] and value['revision']):
            raise ValueError('verified evidence needs source and revision')
    ids = set()
    for record in payload['securities']:
        if set(record) != {'security_id', 'issuer_id', 'exchange', 'instrument_type',
                           'listed_from', 'last_trade_session', 'status', 'source', 'revision',
                           'available_time', 'retrieved_time'}:
            raise ValueError('security schema mismatch')
        provenance(record)
        identity = record['security_id']
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError('unique stable security ID required')
        ids.add(identity)
        if not all(isinstance(record[k], str) and record[k] for k in ('issuer_id', 'exchange')) or record['instrument_type'] not in ('stock', 'etf'):
            raise ValueError('issuer/exchange/instrument type required')
        if record['status'] not in ('active', 'delisted', 'unknown'):
            raise ValueError('explicit lifecycle status required')
        if record['status'] == 'active' and record['last_trade_session'] is not None:
            raise ValueError('active security cannot have a terminal trading date')
        start = day(record['listed_from'])
        if record['last_trade_session'] is not None and day(record['last_trade_session']) < start:
            raise ValueError('last trade precedes listing')
    if not ids:
        raise ValueError('empty security master')
    schemas = {
        'aliases': COMMON | {'ticker', 'effective_from'},
        'membership': COMMON | {'universe', 'member', 'effective_from'},
        'terminals': COMMON | {'last_trade_session', 'settlement_time', 'kind',
                               'reference_price', 'value', 'currency', 'verified', 'basis'},
    }
    for table, fields in schemas.items():
        seen = set()
        identities = {}
        for record in payload[table]:
            if set(record) != fields or record['security_id'] not in ids:
                raise ValueError(table+' schema/identity mismatch')
            provenance(record)
            if not isinstance(record['event_id'], str) or not record['event_id']:
                raise ValueError('event ID required')
            key = (record['event_id'], timestamp(record['available_time']))
            if key in seen:
                raise ValueError('ambiguous duplicate event version')
            seen.add(key)
            if identities.setdefault(record['event_id'], record['security_id']) != record['security_id']:
                raise ValueError('event identity cannot move between securities')
            if table != 'terminals':
                day(record['effective_from'])
            if table == 'aliases' and (not isinstance(record['ticker'], str) or not record['ticker']):
                raise ValueError('ticker required')
            if table == 'membership':
                if record['universe'] != universe['name'] or type(record['member']) is not bool:
                    raise ValueError('invalid membership state/universe')
            if table == 'terminals':
                day(record['last_trade_session'])
                if timestamp(record['settlement_time']).date() < date.fromisoformat(record['last_trade_session']):
                    raise ValueError('settlement precedes last trade')
                if record['kind'] not in ('cash_per_share', 'delisting_return', 'unresolved', 'non_cash'):
                    raise ValueError('unsupported terminal outcome')
                if record['currency'] != 'USD' or type(record['verified']) is not bool:
                    raise ValueError('explicit USD terminal verification required')
                if record['basis'] != 'additional_to_last_close':
                    raise ValueError('terminal basis must exclude already-booked bar cash flows')
                if record['kind'] in ('cash_per_share', 'delisting_return') or record['reference_price'] is not None:
                    finite(record['reference_price'], 0.)
                    if record['reference_price'] <= 0:
                        raise ValueError('positive last-close reference required')
                if record['kind'] in ('unresolved', 'non_cash'):
                    if record['value'] is not None:
                        raise ValueError('unresolved/non-cash outcomes cannot invent a cash value')
                else:
                    finite(record['value'], 0. if record['kind'] == 'cash_per_share' else -1.)
    prices = {}
    seen_prices = set()
    for record in payload['prices']:
        if set(record) != {'security_id', 'bar'} or record['security_id'] not in ids:
            raise ValueError('price identity mismatch')
        bar = Bar(**record['bar'])
        key = (record['security_id'], bar.session)
        if key in seen_prices:
            raise ValueError('duplicate security/session price, including ticker changes')
        seen_prices.add(key)
        if bar.adjustments != 'raw_with_actions':
            raise ValueError('historical prices require raw_with_actions and audited corporate actions')
        prices.setdefault(record['security_id'], []).append(bar)
    for bars in prices.values():
        # Canonical validation is per security, so recycled tickers stay separate.
        validate_bars(sorted(bars, key=lambda b: b.session))
    return payload

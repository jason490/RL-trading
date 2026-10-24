"""Publication-gated source snapshots; not an approved stock-price feature join."""
from copy import deepcopy
from datetime import date, datetime, timezone
import math
import re

from rl_stocks.data.providers.sec.lineage import conservative_availability
from rl_stocks.features.fundamentals.snapshot import NAMES

MAX_AGE = {'10-K': 450, '10-K/A': 450, '10-Q': 180, '10-Q/A': 180}


def _time(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('aware financial decision timestamp required')
    return result.astimezone(timezone.utc)


class ResearchFinancialHistory:
    def __init__(self, entries):
        self._entries = []
        accessions = set()
        for entry in entries:
            snapshot = deepcopy(entry['snapshot'])
            if (snapshot.get('schema') != 'filing-financial-snapshot-v1'
                    or snapshot.get('names') != list(NAMES) or snapshot.get('form') not in MAX_AGE
                    or type(snapshot.get('arithmetic_ready')) is not bool):
                raise ValueError('financial snapshot schema mismatch')
            if (not re.fullmatch(r'\d{10}-\d{2}-\d{6}', snapshot['accession'])
                    or not re.fullmatch(r'[1-9][0-9]{0,9}', snapshot['cik'])
                    or not re.fullmatch(r'[a-f0-9]{64}', snapshot['source_sha256'])
                    or not re.fullmatch(r'[a-f0-9]{64}', entry['index_source_sha256'])):
                raise ValueError('source accession/issuer/checksums required')
            if snapshot['accession'] in accessions:
                raise ValueError('duplicate accession; retain a single immutable source version')
            accessions.add(snapshot['accession'])
            end = date.fromisoformat(snapshot['report_end'])
            filed = date.fromisoformat(snapshot['filed'])
            if end > filed:
                raise ValueError('report period after filing')
            values = snapshot['values']
            if snapshot['arithmetic_ready']:
                if (not isinstance(values, list) or len(values) != len(NAMES)
                        or any(isinstance(v, bool) or not isinstance(v, (float, int))
                               or not math.isfinite(v) for v in values)):
                    raise ValueError('finite ready financial values required')
            elif values is not None:
                raise ValueError('unready snapshot must not contain feature values')
            availability = conservative_availability(snapshot['filed'], entry['accepted_source'])
            self._entries.append(dict(snapshot=snapshot, report_end=end,
                                      available=_time(availability['available_time']),
                                      availability=availability,
                                      index_source_sha256=entry['index_source_sha256']))

    def at(self, cik, decision_time):
        decision = _time(decision_time)
        if decision.year >= 2024:
            raise ValueError('retired decision period excluded from financial research reader')
        eligible = [r for r in self._entries if r['snapshot']['cik'] == cik and r['available'] <= decision]
        if not eligible:
            return dict(source_ready=False, reason='no_available_issuer_snapshot', values=None,
                        financial_features_enabled=False)
        rank = max((r['report_end'], r['available']) for r in eligible)
        selected = [r for r in eligible if (r['report_end'], r['available']) == rank]
        if len(selected) != 1:
            raise ValueError('ambiguous equally ranked accession snapshots')
        record = selected[0]
        snapshot = record['snapshot']
        age = (decision.date() - record['report_end']).days
        reason = None
        if not snapshot['arithmetic_ready']:
            reason = 'latest_snapshot_arithmetic_unready'
        elif age > MAX_AGE[snapshot['form']]:
            reason = 'stale_reporting_period'
        return dict(source_ready=reason is None, reason=reason,
                    values=deepcopy(snapshot['values']) if reason is None else None,
                    accession=snapshot['accession'], cik=cik, report_end=snapshot['report_end'],
                    available_time=record['availability']['available_time'], report_age_days=age,
                    availability_policy=record['availability']['policy'],
                    exact_first_publication_verified=False,
                    source_sha256=snapshot['source_sha256'],
                    index_source_sha256=record['index_source_sha256'],
                    historical_price_link_verified=False, financial_features_enabled=False)

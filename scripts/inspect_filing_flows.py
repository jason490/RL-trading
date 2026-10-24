"""Fixed five-original flow arithmetic and table-presence review; no price access."""
import hashlib
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.features.fundamentals.snapshot import REVENUE, build_flow_snapshot, build_snapshot

SOURCES = ('ibm-early-original-money-facts', 'kr-early-original-money-facts',
           'ibm-original-money-facts', 'kr-original-money-facts',
           'intc-missing-liabilities-original-money-facts')


def main():
    protocol = Path('configs/experiments/filing_flow_feature_rules.json')
    records = []
    for name in SOURCES:
        path = Path('artifacts/filings') / (name + '.json')
        raw = path.read_bytes()
        payload = json.loads(raw)
        snapshot = build_flow_snapshot(payload, '10-K')
        legacy = build_snapshot(payload, '10-K')
        checked = False
        if snapshot['arithmetic_ready']:
            def amount(concepts, interval):
                values = {Fraction(Decimal(f['value'])) for f in payload['facts']
                          if not f['exclusion_reasons'] and f['period_kind'] == 'duration'
                          and [f['start'], f['end']] == interval
                          and f['concept'].split('}', 1)[1] in concepts}
                assert len(values) == 1
                return values.pop()
            revenue = amount(REVENUE, snapshot['current_interval'])
            prior = amount(REVENUE, snapshot['prior_interval'])
            income = amount({'NetIncomeLoss'}, snapshot['current_interval'])
            assert snapshot['values'] == [float(revenue / prior - 1), float(income / revenue)]
            if legacy['arithmetic_ready']:
                assert snapshot['values'] == legacy['values'][:2]
            checked = True
        records.append(dict(payload_sha256=hashlib.sha256(raw).hexdigest(),
                            snapshot=snapshot, legacy_arithmetic_ready=legacy['arithmetic_ready'],
                            independent_fraction_arithmetic_checked=checked))
    presence_path = Path('artifacts/filings/concept-presence-records-v1.json')
    presence = json.loads(presence_path.read_text())
    rows = presence['records']
    count = sum(bool(r['current_total_revenue_present'] and r['current_net_income_present']
                     and r['same_flow_qtrs']) for r in rows)
    result = dict(schema='filing-flow-feasibility-inspection-v1',
                  protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                  table_records_sha256=hashlib.sha256(presence_path.read_bytes()).hexdigest(),
                  records=records, assessed_financial_pairs=len(rows),
                  table_current_flow_presence_pairs=count,
                  table_current_flow_presence_is_original_readiness=False,
                  unassessed_master_pairs=presence['unassessed_master_pairs'],
                  five_originals_are_representative=False, financial_inputs_enabled=False,
                  normalization_fitted=False, training_performed=False, colab_allocated=False)
    path = Path('results/filing-source-review-2026-10-24/flow-feasibility-inspection.json')
    encoded = (json.dumps(result, indent=2) + '\n').encode()
    if path.exists():
        assert path.read_bytes() == encoded
    else:
        write_once(path, encoded)
    print('Originals ready:', sum(r['snapshot']['arithmetic_ready'] for r in records), '/', len(records))
    print('Table current-flow presence only:', count, '/', len(rows))


if __name__ == '__main__':
    main()

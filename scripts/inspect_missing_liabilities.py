"""Reproduce the fixed original-source probe from retained local data; no network."""
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.facts import extract_facts
from rl_stocks.features.fundamentals.snapshot import build_snapshot


def main():
    protocol = Path('configs/experiments/filing_missing_liabilities_probe.json')
    spec = json.loads(protocol.read_text())
    presence_path = Path('artifacts/filings/concept-presence-records-v1.json')
    ledger_path = Path('artifacts/filings/original-candidate-ledger.json')
    presence = json.loads(presence_path.read_text())
    ledger = json.loads(ledger_path.read_text())
    by_key = {(r['cik'], r['accession']): r for r in ledger['records']}
    candidates = [r for r in presence['records'] if r['filing_year'] == '2020'
                  and r['form'] == '10-K' and not r['current_liabilities_present']
                  and all(r[k] for k in ('current_assets_present', 'current_net_income_present',
                                        'current_total_revenue_present'))]
    candidates.sort(key=lambda r: (by_key[r['cik'], r['accession']]['archive_filed'], r['accession']))
    assert len(candidates) == spec['eligible_count']
    chosen = candidates[0]
    assert (chosen['accession'], chosen['cik']) == (spec['accession'], spec['cik'])
    metadata = by_key[chosen['cik'], chosen['accession']]
    assert metadata['source_url'] == spec['source_url']
    sha = 'b7b59760abcb0026c90814d3a5709ca8e26a02a38e56a018e21de025b9193fa7'
    source = Path('data/raw/sec-xbrl') / sha / 'instance.xml'
    raw = source.read_bytes()
    assert len(raw) <= 20_000_000 and hashlib.sha256(raw).hexdigest() == sha
    # The extractor subsequently validates namespace, issuer, periods and all contexts.
    root = ET.fromstring(raw)
    ends = {n.text for n in root if n.tag.endswith('}DocumentPeriodEndDate')}
    assert len(ends) == 1
    report_end = ends.pop()
    payload = extract_facts(source, sha, chosen['accession'], chosen['cik'],
                            report_end, '2020-01-24')
    snapshot = build_snapshot(payload, chosen['form'])
    current = sorted({f['concept'].split('}', 1)[1] for f in payload['facts']
                      if not f['exclusion_reasons'] and f['period_kind'] == 'instant'
                      and f['end'] == report_end})
    result = dict(schema='original-missing-liabilities-probe-result-v1',
                  protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                  concept_records_sha256=hashlib.sha256(presence_path.read_bytes()).hexdigest(),
                  candidate_ledger_sha256=hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                  accession=chosen['accession'], cik=chosen['cik'],
                  source_url=spec['source_url'], source_sha256=sha, source_bytes=len(raw),
                  archive_period_label=metadata['archive_period_label'],
                  original_report_end=report_end, original_current_balance_concepts=current,
                  original_current_direct_liabilities_present='Liabilities' in current,
                  snapshot=snapshot, source_counts=payload['counts'],
                  sample_is_representative=False, liability_imputation_permitted=False,
                  financial_inputs_enabled=False, training_performed=False, colab_allocated=False)
    output = Path('results/filing-source-review-2026-10-24/missing-liabilities-probe.json')
    encoded = (json.dumps(result, indent=2) + '\n').encode()
    if output.exists():
        assert output.read_bytes() == encoded
    else:
        write_once(output, encoded)
    print('Fixed probe reproduced:', chosen['accession'], report_end,
          'direct liabilities:', 'Liabilities' in current, 'snapshot:', snapshot['reason'])


if __name__ == '__main__':
    main()

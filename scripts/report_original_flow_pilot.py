"""Verify retained pilot originals and report the fixed source-feasibility gates."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.facts import extract_facts
from rl_stocks.features.fundamentals.snapshot import build_flow_snapshot

STATE = Path('.runtime/sec-original-flow-pilot.json')
MANIFEST = Path('artifacts/filings/original-flow-pilot-manifest.json')
PROTOCOL = Path('configs/experiments/filing_original_flow_pilot.json')
RULES = Path('configs/experiments/filing_flow_feature_rules.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(state, manifest):
    if state['status'] not in {'complete', 'failed', 'interrupted'}:
        raise ValueError('terminal collection state required; no live-result freezing')
    selected = manifest['selected']
    def key(row):
        return row['cik'], row['accession'], row['stratum']
    completed = state['completed']
    if [key(r) for r in completed] != [key(r) for r in selected[:len(completed)]]:
        raise ValueError('completed source records differ from fixed selection/order')
    if len(completed) > len(selected) or (state['status'] == 'complete' and len(completed) != len(selected)):
        raise ValueError('collection completeness mismatch')
    for row in completed:
        if row['status'] not in {'inspected', 'unsupported_original', 'source_missing_404'}:
            raise ValueError('unsupported collection result status')
        expected_ready = row['snapshot']['arithmetic_ready'] if row['status'] == 'inspected' else False
        if row['arithmetic_ready'] is not expected_ready:
            raise ValueError('source readiness flag differs from inspected snapshot')
    totals = Counter(r['stratum'] for r in selected)
    ready = Counter(r['stratum'] for r in completed if r['arithmetic_ready'])
    absent = Counter(r['stratum'] for r in manifest['absent_issuer_strata'])
    evaluated = state['status'] == 'complete'
    strata = {label: dict(selected=total, arithmetic_ready=ready[label],
                          absent_issuer_slots=absent[label],
                          readiness_fraction=ready[label] / total if evaluated else None,
                          threshold_pass=(10 * ready[label] >= 7 * total) if evaluated else None)
              for label, total in sorted(totals.items())}
    count = sum(ready.values())
    passed = evaluated and 5 * count >= 4 * len(selected) and all(
        value['threshold_pass'] for value in strata.values())
    return dict(selected_originals=len(selected), completed_originals=len(completed),
                candidate_ciks=len(manifest['candidate_ciks']),
                absent_issuer_strata=len(manifest['absent_issuer_strata']),
                arithmetic_ready=count, readiness_fraction=count / len(selected) if evaluated else None,
                per_stratum=strata, status_counts=dict(Counter(r['status'] for r in completed)),
                unready_reasons=dict(Counter(r['snapshot']['reason'] for r in completed
                                             if r['status'] == 'inspected' and not r['arithmetic_ready'])),
                unsupported_error_types=dict(Counter(r['error_type'] for r in completed
                                                       if r['status'] == 'unsupported_original')),
                archive_period_label_differences=sum(r.get('archive_period_differs', False) for r in completed),
                gate_evaluated=evaluated, fixed_source_feasibility_gate_passed=passed,
                incomplete_selected_sources=len(selected) - len(completed))


def main():
    state = json.loads(STATE.read_text())
    manifest = json.loads(MANIFEST.read_text())
    if (state['manifest_sha256'] != sha(MANIFEST)
            or manifest['protocol_sha256'] != sha(PROTOCOL)
            or manifest['rules_sha256'] != sha(RULES)
            or manifest['ledger_sha256'] != sha(Path('artifacts/filings/original-candidate-ledger.json'))):
        raise ValueError('fixed pilot input hash mismatch')
    summary = summarize(state, manifest)
    metadata = {(r['cik'], r['accession']): r for r in manifest['selected']}
    verified_bytes = 0
    for record in state['completed']:
        if record['status'] == 'source_missing_404':
            continue
        path = Path(record['source_path'])
        if sha(path) != record['source_sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError('retained original source checksum/size mismatch')
        verified_bytes += record['bytes']
        if record['status'] != 'inspected':
            continue
        facts_path = Path(record['fact_payload_path'])
        if sha(facts_path) != record['fact_payload_sha256']:
            raise ValueError('retained fact payload checksum mismatch')
        facts = json.loads(facts_path.read_text())
        expected = metadata[record['cik'], record['accession']]
        filed = expected['archive_filed']
        filed = f'{filed[:4]}-{filed[4:6]}-{filed[6:]}'
        reproduced = extract_facts(path, record['source_sha256'], record['accession'], record['cik'],
                                   record['original_report_end'], filed)
        if reproduced != facts or build_flow_snapshot(reproduced, expected['form']) != record['snapshot']:
            raise ValueError('original facts or source arithmetic do not reproduce')
    if verified_bytes != state['total_source_bytes']:
        raise ValueError('retained byte accounting mismatch')
    result = dict(schema='sec-original-flow-pilot-summary-v1', collection_status=state['status'],
                  source_git_revision=state['source_git_revision'],
                  terminal_error_type=state.get('error_type'),
                  unfinished_current_accession=state.get('current_accession'),
                  state_sha256=sha(STATE), manifest_sha256=sha(MANIFEST),
                  protocol_sha256=sha(PROTOCOL), rules_sha256=sha(RULES),
                  total_verified_source_bytes=verified_bytes,
                  elapsed_seconds=state['elapsed_seconds'], **summary,
                  all_completed_source_hashes_verified=True,
                  inspected_fact_extractions_and_snapshots_reproduced=True,
                  original_vintage_verified=False, historical_price_links_verified=False,
                  financial_inputs_enabled=False, normalization_fitted=False,
                  training_performed=False, colab_allocated=False)
    output = Path('results/filing-source-review-2026-10-24/original-flow-pilot-summary.json')
    encoded = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
    if output.exists():
        if output.read_bytes() != encoded:
            raise ValueError('immutable pilot summary differs')
    else:
        write_once(output, encoded)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

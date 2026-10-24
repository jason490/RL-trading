"""Prepare and collect the fixed cross-era original-filing source pilot."""
import argparse
import fcntl
import hashlib
import json
import shutil
import signal
import time
from pathlib import Path
from xml.etree import ElementTree as ET

from collect_sec_archives import _save
from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.download import identity
from rl_stocks.data.providers.sec.facts import extract_facts
from rl_stocks.data.providers.sec.originals import OriginalMissing, acquire_original
from rl_stocks.features.fundamentals.snapshot import build_flow_snapshot

PROTOCOL = Path('configs/experiments/filing_original_flow_pilot.json')
RULES = Path('configs/experiments/filing_flow_feature_rules.json')
LEDGER = Path('artifacts/filings/original-candidate-ledger.json')
MANIFEST = Path('artifacts/filings/original-flow-pilot-manifest.json')
STATE = Path('.runtime/sec-original-flow-pilot.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(ledger, strata, candidate_ciks):
    records = ledger['records']
    selected, absent = [], []
    for cik in sorted(set(candidate_ciks), key=int):
        for first, last in strata:
            candidates = [r for r in records if r['cik'] == cik and r['form'] == '10-K'
                          and first <= int(r['archive_filed'][:4]) <= last]
            label = f'{first}-{last}'
            if not candidates:
                absent.append(dict(cik=cik, stratum=label))
                continue
            chosen = min(candidates, key=lambda r: (r['archive_filed'], r['accession']))
            selected.append(dict(chosen, stratum=label))
    return selected, absent


def prepare():
    spec, ledger = json.loads(PROTOCOL.read_text()), json.loads(LEDGER.read_text())
    if ledger['candidate_cik_count'] != 112 or ledger['accession_count'] != 6161:
        raise ValueError('fixed original source-candidate ledger required')
    discovery_path = Path('artifacts/filings/current-identity-discovery.json')
    cohort_path = Path('configs/filings/cohort_discovery_v1.json')
    discovery = json.loads(discovery_path.read_text())
    cohort = json.loads(cohort_path.read_text())
    ciks = {c['cik'] for candidates in discovery['candidates'].values() for c in candidates}
    ciks.update(c['cik'] for c in cohort['additional_candidates'])
    if len(ciks) != 112:
        raise ValueError('all112 source candidate CIKs required, including absent issuers')
    selected, absent = select(ledger, spec['strata'], ciks)
    result = dict(schema='sec-original-flow-pilot-manifest-v1',
                  protocol_sha256=digest(PROTOCOL), rules_sha256=digest(RULES),
                  ledger_sha256=digest(LEDGER), discovery_sha256=digest(discovery_path),
                  cohort_sha256=digest(cohort_path), candidate_ciks=sorted(ciks, key=int),
                  selected=selected, absent_issuer_strata=absent,
                  financial_inputs_enabled=False, training_performed=False)
    write_once(MANIFEST, (json.dumps(result, indent=2) + '\n').encode())
    print('Prepared selected:', len(selected), 'absent issuer/strata:', len(absent), flush=True)
    return result


def inspect(path, record, output_root):
    raw = path.read_bytes()
    text = raw.decode('utf-8-sig')
    if '<!DOCTYPE' in text.upper() or '<!ENTITY' in text.upper():
        raise ValueError('unsupported DTD/entity source')
    root = ET.fromstring(text)
    ends = {n.text for n in root if n.tag.endswith('}DocumentPeriodEndDate')}
    if len(ends) != 1:
        raise ValueError('unique original report end required')
    end = ends.pop()
    filed = record['archive_filed']
    filed = f'{filed[:4]}-{filed[4:6]}-{filed[6:]}'
    payload = extract_facts(path, path.parent.name, record['accession'], record['cik'], end, filed)
    output = output_root / (record['accession'] + '-facts.json')
    write_once(output, (json.dumps(payload, indent=2) + '\n').encode())
    snapshot = build_flow_snapshot(payload, record['form'])
    return dict(original_report_end=end, snapshot=snapshot,
                fact_payload_path=str(output), fact_payload_sha256=digest(output),
                archive_period_differs=end.replace('-', '') != record['archive_period_label'])


def collect():
    manifest = prepare()
    spec = json.loads(PROTOCOL.read_text())
    root = Path('data/raw/sec-xbrl')
    root.mkdir(parents=True, exist_ok=True)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with STATE.with_suffix('.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if STATE.exists():
            raise ValueError('existing collection requires review; automatic restart disabled')
        user_agent = identity()
        started = time.monotonic()
        state = dict(schema='sec-original-flow-pilot-state-v1', status='running',
                     manifest_sha256=digest(MANIFEST), started_utc_epoch=time.time(),
                     source_git_revision=None, completed=[], total_source_bytes=0,
                     current_accession=None, elapsed_seconds=0, financial_inputs_enabled=False,
                     training_performed=False, colab_allocated=False)
        # Source revision is recorded without environment/credential output.
        import subprocess
        state['source_git_revision'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip()
        _save(STATE, state)
        try:
            for record in manifest['selected']:
                if time.monotonic() - started + spec['next_operation_reserve_seconds'] > spec['maximum_collection_seconds']:
                    raise TimeoutError('pilot operation reserve exceeds deadline')
                if state['total_source_bytes'] + spec['maximum_source_bytes'] > spec['maximum_total_source_bytes']:
                    raise ValueError('pilot byte reserve exhausted')
                if shutil.disk_usage(root).free < spec['minimum_free_disk_bytes'] + spec['maximum_source_bytes']:
                    raise ValueError('pilot free-disk reserve exhausted')
                state['current_accession'] = record['accession']
                _save(STATE, state)
                result = dict(accession=record['accession'], cik=record['cik'], stratum=record['stratum'])
                try:
                    path = acquire_original(root, record, user_agent)
                except OriginalMissing:
                    result.update(status='source_missing_404', arithmetic_ready=False)
                else:
                    size = path.stat().st_size
                    state['total_source_bytes'] += size
                    result.update(source_path=str(path), source_sha256=path.parent.name, bytes=size)
                    try:
                        inspected = inspect(path, record, Path('artifacts/filings/original-flow-pilot'))
                    except (ValueError, ET.ParseError, UnicodeError, ArithmeticError) as error:
                        result.update(status='unsupported_original', arithmetic_ready=False,
                                      error_type=type(error).__name__)
                    else:
                        result.update(inspected, status='inspected',
                                      arithmetic_ready=inspected['snapshot']['arithmetic_ready'])
                state['completed'].append(result)
                state['elapsed_seconds'] = time.monotonic() - started
                _save(STATE, state)
                print(f"Inspected {len(state['completed'])}/{len(manifest['selected'])}: "
                      f"{record['accession']} {result['status']} ready={result['arithmetic_ready']}", flush=True)
            state['status'] = 'complete'
            state['current_accession'] = None
        except BaseException as error:
            state['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
            state['error_type'] = type(error).__name__
            raise
        finally:
            state['elapsed_seconds'] = time.monotonic() - started
            _save(STATE, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
    else:
        def terminate(signum, frame):
            raise KeyboardInterrupt('pilot termination requested')
        signal.signal(signal.SIGTERM, terminate)
        collect()


if __name__ == '__main__':
    main()

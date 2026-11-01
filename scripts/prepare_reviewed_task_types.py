"""Compile feature-scoped initial type evidence for the fixed research dataset."""
import argparse
import json
from pathlib import Path
import re

from rl_stocks.core.contracts import digest
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.data.universe.type_contract import verification_scope
from rl_stocks.inference.bundle import checksum
from rl_stocks.training.task_conditioning import validate_type_manifest
from review_task_type_covers import normalized_text

ROOT = Path('results/task-conditioning-2026-10-31')
EVENT_RULES = {
    'GOOGL': ('Alphabet Class A Common Stock', 'Alphabet Class C Capital Stock',
              'automatically converted into an equivalent corresponding share of Alphabet stock'),
    'LIN': ('share of common stock of Praxair', 'converted into the right to receive one ordinary share',
            'of Linde plc'),
}


def event_reviews(protected):
    source = Path('configs/filings/issuer_evidence_v1.json')
    protected[str(source)] = checksum(source)
    events = json.loads(source.read_text())['records']
    result = {}
    for symbol, anchors in EVENT_RULES.items():
        event = next(r for r in events if r['symbol'] == symbol)
        path = Path('data/raw/sec-filings') / event['source_sha256'] / 'filing.htm'
        if checksum(path) != event['source_sha256']:
            raise ValueError('original equity-preserving event changed')
        protected[str(path)] = event['source_sha256']
        text = normalized_text(path)
        if any(anchor not in text for anchor in anchors):
            raise ValueError('reviewed corporate-equity conversion unsupported')
        result[symbol] = {'kind_before': 'stock', 'kind_after': 'stock',
            'source_url': event['source_url'], 'source_sha256': event['source_sha256'],
            'event_available_time': event['available_time'],
            'initial_type_value_taken_from_event': False,
            'historical_price_link_verified': False, 'financial_comparability_verified': False}
    return result


def qqq_review(protected):
    collection_path = ROOT / 'fund-source-collection.json'
    later_path = ROOT / 'qqq-alias-source-collection.json'
    protected[str(collection_path)], protected[str(later_path)] = checksum(collection_path), checksum(later_path)
    initial = next(r for r in json.loads(collection_path.read_text())['cases'] if r['symbol'] == 'QQQ')
    later = json.loads(later_path.read_text())
    if later['status'] != 'complete' or later['master_record']['cik'] != '1067839':
        raise ValueError('original same-fund registration required')
    cusips = []
    for item in (initial['primary'], later['primary']):
        if checksum(item['path']) != item['receipt']['sha256']:
            raise ValueError('original fund registration changed')
        protected[item['path']] = item['receipt']['sha256']
        text = normalized_text(item['path'])
        if not re.search(r'The Trust, an exchange.traded fund', text, re.I):
            raise ValueError('explicit named-trust ETF definition required')
        match = re.search(r'CUSIP:\s*((?:[0-9A-Z]\s*){9})', text)
        if match is None:
            raise ValueError('original fund identifier required')
        cusips.append(re.sub(r'\s+', '', match[1]))
    if cusips != ['73935A104', '73935A104']:
        raise ValueError('same named fund identifier required')
    return {'initial_kind': 'etf', 'later_kind': 'etf', 'fund_registrant_cik': '1067839',
            'cusip': cusips[0], 'exact_ticker_change_date_verified': False,
            'later_document_still_uses_qqqq': True,
            'binding_scope': 'Named fund subject under existing dataset provider association',
            'initial_type_value_taken_from_later_document': False,
            'historical_price_link_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.exists() or args.review.exists():
        raise ValueError('fresh deterministic manifest/review outputs required')
    ledger_path = ROOT / 'initial-type-source-ledger.json'
    ledger = json.loads(ledger_path.read_text())
    dataset_id = 'cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b'
    folder = Path('data/datasets') / dataset_id
    protected = {str(p): checksum(p) for p in
                 (ledger_path, folder / 'dataset.json', Path(__file__),
                  Path('src/rl_stocks/data/universe/type_contract.py'),
                  Path('scripts/review_task_type_covers.py'), Path('.runtime/usage-ledger.json'))}
    # Independent source facts were verified before this association review.
    for path, sha in ledger['protected_sha256'].items():
        if checksum(path) != sha:
            raise ValueError('initial source fact input changed')
        protected[path] = sha
    events = event_reviews(protected)
    fund = qqq_review(protected)
    records = []
    bindings = {}
    for row in ledger['records']:
        symbol, kind = row['symbol'], row['kind_supported_by_source']
        if not row['initial_type_fact_supported']:
            raise ValueError('initial public type fact required; no snapshot fallback')
        if symbol in events:
            if kind != events[symbol]['kind_before'] or kind != events[symbol]['kind_after']:
                raise ValueError('mixed-kind identifier path rejects fixed-type research')
            binding = 'corporate_equity_preserving_identifier_event'
        elif symbol == 'QQQ':
            if kind != fund['initial_kind'] or kind != fund['later_kind']:
                raise ValueError('mixed-kind named-fund path rejected')
            binding = 'same_named_fund; provider_canonical_identifier_assumed'
        elif symbol == 'IWM':
            binding = 'named_ETF_underlying_in_contemporaneous_legal_contract; no_fund_issuer_join'
        elif row['explicit_canonical_ticker_observation']:
            binding = 'contemporaneous_canonical_ticker_and_type_observation'
        elif row['same_cik_in_current_discovery'] and kind == 'stock':
            binding = 'dated_corporate_equity_for_same_issuer_id; provider_series_association_assumed'
        else:
            raise ValueError('unsupported coarse-type subject association')
        bindings[symbol] = binding
        available = row['publication']['available_time']
        records.append({'symbol': symbol, 'kind': kind, 'source_fact_kind': kind,
                        'known_from': available[:10], 'source_available_time': available,
                        'source': row['source_url'], 'source_sha256': row['source_sha256'],
                        'evidence_kind': 'historical_primary_source',
                        'binding_scope': binding, 'historical_price_link_verified': False})
    manifest = {'schema': 'historical-instrument-types-v1', 'dataset': dataset_id,
                'historical_evidence_verified': True, 'verification_scope': verification_scope(),
                'records': records,
                'assumptions': ['Use existing cached provider series association; no new price/issuer join',
                    'Carry initial coarse kind under fixed-kind research contract; reject evidenced mixed-kind paths',
                    'Two-XNYS-session availability reconstruction is not exact dissemination verification'],
                'financial_features_enabled': False, 'sector_snapshot_inputs_enabled': False,
                'historical_universe_membership_verified': False,
                'paper_execution_eligible': False}
    data = read_dataset(folder)
    identity = validate_type_manifest(data, manifest)
    if len(records) != 111 or digest(data['normalizer']) != 'b3c8a298d66f253e4e4f5474214873d6b6aae16d1b4666633c6d83be6041469f':
        raise ValueError('fixed111 source and original normalization required')
    if not all(checksum(path) == sha for path, sha in protected.items()):
        raise ValueError('protected source changed')
    review = {'schema': 'task-coarse-type-admission-review-v1', 'manifest_sha256': identity,
              'source_fact_ledger_sha256': checksum(ledger_path), 'binding_by_symbol': bindings,
              'equity_preserving_events': events, 'named_fund_review': fund,
              'verification_scope': verification_scope(),
              'admitted_feature': 'initial_coarse_stock_etf_type_only',
              'records': len(records), 'protected_sha256': protected,
              'protected_sources_unchanged': True, 'price_identity_verified': False,
              'historical_membership_verified': False, 'financial_features_enabled': False,
              'model_performance_verified': False, 'paper_execution_eligible': False,
              'training_performed': False, 'colab_allocated': False,
              'limitation': 'Conditional on the existing provider instrument association; exact ticker intervals/share classes and financial price joins remain unverified'}
    write_once(args.manifest, (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode())
    write_once(args.review, (json.dumps(review, indent=2, allow_nan=False) + '\n').encode())
    print('Admitted111 initial coarse-type research records:', identity)
    print('Price identity/financial joins/paper eligibility remain unverified; no model training')


if __name__ == '__main__':
    main()

"""Verify the three fixed original-cover observations against their retained bytes."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re

from rl_stocks.core.files import write_once
from rl_stocks.data.providers.sec.lineage import conservative_availability
from rl_stocks.inference.bundle import checksum

# Brief source anchors, not a general automatic ticker classifier.
OBSERVATIONS = {
    'ABT': {'security_anchor': 'Common Shares, Without Par Value',
            'issuer_anchor': 'Abbott Laboratories', 'ticker_anchor': None},
    'ADBE': {'security_anchor': 'common stock is traded on the NASDAQ Global Select Market under the symbol “ADBE.”',
             'issuer_anchor': 'ADOBE SYSTEMS INCORPORATED',
             'ticker_anchor': 'under the symbol “ADBE.”'},
    'AEP': {'security_anchor': 'American Electric Power Company, Inc. Common Stock, $6.50 par value New York Stock Exchange',
            'issuer_anchor': 'American Electric Power Company, Inc.', 'ticker_anchor': None},
}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skipped = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skipped += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skipped = max(0, self.skipped - 1)

    def handle_data(self, data):
        if not self.skipped:
            self.parts.append(data)


def normalized_text(path):
    parser = _Text()
    parser.feed(Path(path).read_bytes().decode('utf-8-sig'))
    return ' '.join(' '.join(parser.parts).split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh original-cover review output required')
    source = Path('results/task-conditioning-2026-10-31/original-cover-collection.json')
    collected = json.loads(source.read_text())
    if collected['status'] != 'complete' or {r['symbol'] for r in collected['cases']} != set(OBSERVATIONS):
        raise ValueError('complete original fixed three-source collection required')
    protected = {str(p): checksum(p) for p in
                 (source, Path('.runtime/usage-ledger.json'), Path(__file__))}
    records = []
    for case in collected['cases']:
        for name in ('index', 'primary'):
            item = case[name]
            path = Path(item['path'])
            if checksum(path) != item['receipt']['sha256']:
                raise ValueError('original collected source changed')
            protected[str(path)] = item['receipt']['sha256']
        text = normalized_text(case['primary']['path'])
        index = normalized_text(case['index']['path'])
        observation = OBSERVATIONS[case['symbol']]
        for anchor in observation.values():
            if anchor is not None and anchor not in text:
                raise ValueError('reviewed source anchor missing; do not substitute')
        filed = re.findall(r'Filing Date (\d{4}-\d{2}-\d{2})', index)
        accepted = re.findall(r'Accepted (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', index)
        if len(filed) != 1 or len(accepted) != 1:
            raise ValueError('unambiguous original filing/acceptance metadata required')
        availability = conservative_availability(filed[0], accepted[0])
        if availability['available_time'][:10] > case['first_training_session']:
            raise ValueError('conservative availability too late for initial training row')
        records.append({'symbol': case['symbol'], 'cik': case['cik'],
                        'accession': case['accession'], 'source_url': case['primary']['receipt']['source_url'],
                        'source_sha256': case['primary']['receipt']['sha256'],
                        'filed': filed[0], 'accepted_source': accepted[0],
                        'availability': availability, 'observation': observation,
                        'common_equity_supported_in_document': True,
                        'exact_ticker_clause_supported': observation['ticker_anchor'] is not None,
                        'historical_metadata_admitted': False, 'validity_interval_inferred': False})
    if not all(checksum(Path(p)) == value for p, value in protected.items()):
        raise ValueError('protected review input changed')
    result = {'schema': 'task-original-cover-review-v1', 'records': records,
              'protected_sha256': protected, 'protected_sources_unchanged': True,
              'historical_metadata_admitted': False, 'network_requests': 0,
              'training_performed': False, 'colab_allocated': False}
    write_once(args.output, (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    print('Three original common-equity observations; explicit ticker clause only for ADBE')


if __name__ == '__main__':
    main()

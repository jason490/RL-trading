"""Conservative original-HTML review hints; no automatic historical admission."""
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import re


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('style', 'script'):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ('style', 'script'):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def inspect_html_stock(path, expected_sha256, *, symbol):
    if not re.fullmatch(r'[A-Z][A-Z0-9.-]{0,9}', symbol):
        raise ValueError('explicit canonical ticker required')
    with Path(path).open('rb') as stream:
        raw = stream.read(20_000_001)
    if len(raw) > 20_000_000 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('bounded original HTML checksum mismatch')
    parser = _Text()
    parser.feed(raw.decode('utf-8-sig'))
    text = ' '.join(' '.join(parser.parts).split())
    common = re.compile(r'\b(?:common\s+(?:stock|shares)|ordinary\s+shares)\b', re.I)
    cover = text[:14000]
    registration = re.search(r'\bsecurities\s+registered\b', cover, re.I)
    cover_common = (common.search(cover, registration.start(), registration.start() + 1800)
                    if registration else None)
    ticker = re.compile(r'\b(?i:(?:ticker|trading|stock|market)\s+symbol|symbol)s?\s*'
                        r'(?i:is\s*|of\s*|:\s*)?[“"\'(]?\s*' + re.escape(symbol) + r'\b(?![A-Za-z0-9-]|\.[A-Za-z0-9])')
    clauses = []
    for match in ticker.finditer(text):
        start, end = max(0, match.start() - 400), min(len(text), match.end() + 400)
        stock = common.search(text, start, end)
        if stock is None:
            continue
        words = text[start:end].split()
        # Keep a short review window, never a substitute for the original document.
        relative = text[start:match.start()].split()
        first = max(0, len(relative) - 8)
        excerpt = ' '.join(words[first:first + 18])
        clauses.append({'ticker_anchor': match.group(), 'position': match.start(),
                         'nearby_common_equity': True, 'review_excerpt': excerpt})
    return {'schema': 'sec-html-stock-review-hints-v1', 'source_sha256': expected_sha256,
            'expected_symbol': symbol, 'common_equity_cover_hint': cover_common is not None,
            'explicit_ticker_clause_hint': bool(clauses),
            'first_ticker_clause': clauses[0] if clauses else None,
            'historical_identity_verified': False, 'historical_metadata_admitted': False,
            'validity_interval_inferred': False, 'model_inputs_enabled': False}

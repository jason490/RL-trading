"""Extract dated issuer/ticker evidence; never infer an identity validity interval."""
import hashlib
import re
from html.parser import HTMLParser
from pathlib import Path

CONCEPTS = {
    'dei:entitycentralindexkey', 'dei:tradingsymbol', 'dei:security12btitle',
    'dei:securityexchangename', 'dei:documentperiodenddate', 'dei:documenttype',
}


class _IdentityParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.active = []
        self.facts = []

    def handle_starttag(self, tag, attrs):
        if tag == 'ix:nonnumeric':
            attrs = dict(attrs)
            concept = attrs.get('name', '').lower()
            self.active.append(dict(concept=concept, context=attrs.get('contextref', ''), text=[]))

    def handle_data(self, data):
        for record in self.active:
            record['text'].append(data)

    def handle_endtag(self, tag):
        if tag == 'ix:nonnumeric' and self.active:
            record = self.active.pop()
            if record['concept'] in CONCEPTS:
                self.facts.append(dict(concept=record['concept'], context=record['context'],
                                       value=' '.join(''.join(record['text']).split())))


def inspect_identity(path, expected_sha256, expected_cik, symbol):
    """Report explicit inline facts, including alternative securities/classes.

    Caller must retain accession, official filing date and provenance separately.
    A positive match applies to this document only, never the whole price history.
    """
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('original filing checksum mismatch')
    if (not isinstance(expected_cik, str)
            or not re.fullmatch(r'[0-9]{1,10}', expected_cik) or int(expected_cik) <= 0):
        raise ValueError('valid expected issuer CIK required')
    if not isinstance(symbol, str) or not symbol:
        raise ValueError('expected ticker required')
    parser = _IdentityParser()
    parser.feed(raw.decode('utf-8-sig'))
    parser.close()
    if parser.active:
        raise ValueError('unterminated inline identity fact')
    cik_facts = [r['value'] for r in parser.facts
                 if r['concept'] == 'dei:entitycentralindexkey']
    ciks_valid = bool(cik_facts) and all(
        re.fullmatch(r'[0-9]{1,10}', value) and int(value) > 0 for value in cik_facts)
    ciks = {str(int(value)) for value in cik_facts} if ciks_valid else set()
    symbols = sorted({r['value'] for r in parser.facts if r['concept'] == 'dei:tradingsymbol'})
    matched = ciks == {str(int(expected_cik))} and symbol in symbols
    return dict(schema='sec-filing-identity-evidence-v1', source_sha256=expected_sha256,
                expected_cik=str(int(expected_cik)), expected_symbol=symbol,
                declared_ciks=sorted(ciks), declared_symbols=symbols,
                issuer_and_ticker_match=matched, facts=parser.facts,
                historical_mapping_verified=False, validity_interval_inferred=False,
                model_inputs_enabled=False)

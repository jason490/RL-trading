"""Dated original-instance stock identity hints; never infer a validity interval."""
import hashlib
import io
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from xml.etree import ElementTree as ET

X = '{http://www.xbrl.org/2003/instance}'
DEI = r'\{https?://(?:xbrl.sec.gov|xbrl.us)/dei/[^}]+\}'
FIELDS = ('TradingSymbol', 'EntityRegistrantName', 'DocumentType',
          'EntityCommonStockSharesOutstanding', 'DocumentPeriodEndDate')


def inspect_stock_type(path, expected_sha256, *, cik, symbol, form, filed):
    if (not re.fullmatch(r'[0-9]{1,10}', cik) or int(cik) <= 0
            or not symbol or form not in ('10-K', '10-Q', '10-K/A', '10-Q/A')):
        raise ValueError('explicit stock filing identity required')
    filing = date.fromisoformat(filed)
    if filing >= date(2024, 1, 1):
        raise ValueError('retired filing source rejected')
    with Path(path).open('rb') as stream:
        raw = stream.read(20_000_001)
    if len(raw) > 20_000_000 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('bounded original source checksum mismatch')
    if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError('DTD/entity declarations rejected')
    namespaces = {}
    for _, (prefix, uri) in ET.iterparse(io.BytesIO(raw), events=('start-ns',)):
        if prefix in namespaces and namespaces[prefix] != uri:
            raise ValueError('namespace rebinding rejected')
        namespaces[prefix] = uri
    root = ET.fromstring(raw)
    if root.tag != X + 'xbrl':
        raise ValueError('original XBRL instance required')
    contexts, units = {}, {}
    for tag, index in (('context', contexts), ('unit', units)):
        for node in root.findall(X + tag):
            key = node.get('id')
            if not key or key in index:
                raise ValueError('unique source context/unit IDs required')
            index[key] = node
    facts, excluded = [], []
    for node in root:
        name = next((name for name in FIELDS if re.fullmatch(DEI + name, node.tag)), None)
        if name is None:
            continue
        context = contexts.get(node.get('contextRef'))
        if context is None:
            raise ValueError('orphan selected source fact')
        identifiers = context.findall(X + 'entity/' + X + 'identifier')
        reasons = []
        if (len(identifiers) != 1 or identifiers[0].get('scheme') not in
                ('http://www.sec.gov/CIK', 'https://www.sec.gov/CIK')
                or not re.fullmatch(r'[0-9]{1,10}', identifiers[0].text or '')
                or int(identifiers[0].text) != int(cik)):
            reasons.append('issuer_mismatch')
        if (context.find(X + 'entity/' + X + 'segment') is not None
                or context.find(X + 'scenario') is not None):
            reasons.append('qualified_context')
        period = context.find(X + 'period')
        dates = [n.text for n in period] if period is not None else []
        if (not dates or any(not value or date.fromisoformat(value) > filing for value in dates)):
            reasons.append('invalid_or_late_period')
        if node.get('{http://www.w3.org/2001/XMLSchema-instance}nil', 'false') != 'false':
            reasons.append('nil_or_unsupported_flag')
        text = (node.text or '').strip()
        if name == 'EntityCommonStockSharesOutstanding':
            unit = units.get(node.get('unitRef'))
            measure = unit.findtext(X + 'measure') if unit is not None else None
            prefix, sep, local = (measure or '').partition(':')
            if not sep:
                prefix, local = '', prefix
            if (unit is None or len(unit) != 1 or local != 'shares'
                    or namespaces.get(prefix) != X[1:-1]):
                reasons.append('not_simple_shares')
            try:
                number = Decimal(text)
                if not number.is_finite() or number <= 0:
                    reasons.append('invalid_share_count')
            except InvalidOperation:
                reasons.append('invalid_share_count')
        fact = {'concept': name, 'context': node.get('contextRef')}
        if name != 'EntityCommonStockSharesOutstanding':
            fact['text'] = text  # Share counts are source checks, never model inputs.
        if reasons:
            excluded.append({**fact, 'reasons': reasons})
        else:
            facts.append(fact)
    values = {name: sorted({r.get('text', '') for r in facts if r['concept'] == name})
              for name in FIELDS}
    matched = (symbol in values['TradingSymbol'] and values['DocumentType'] == [form]
               and bool(values['EntityRegistrantName'])
               and any(r['concept'] == 'EntityCommonStockSharesOutstanding' for r in facts))
    return {'schema': 'sec-stock-type-document-v1', 'source_sha256': expected_sha256,
            'expected_cik': str(int(cik)), 'expected_symbol': symbol, 'filed': filed,
            'document_exact_stock_identity_hint': matched, 'facts': facts, 'excluded': excluded,
            'historical_metadata_admitted': False, 'validity_interval_inferred': False,
            'model_inputs_enabled': False}

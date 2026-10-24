"""Restricted original XBRL fact extraction for financial-source review, not inference."""
import hashlib
import io
import re
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from xml.etree import ElementTree as ET

X = '{http://www.xbrl.org/2003/instance}'
NIL = '{http://www.w3.org/2001/XMLSchema-instance}nil'
MONEY_CONCEPTS = frozenset({
    'Assets', 'Liabilities', 'NetIncomeLoss', 'Revenues',
    'RevenueFromContractWithCustomerExcludingAssessedTax',
    'RevenueFromContractWithCustomerIncludingAssessedTax',
    'SalesRevenueNet', 'SalesRevenueGoodsNet', 'SalesRevenueServicesNet',
})


BALANCE_CONCEPTS = frozenset({
    'Assets', 'Liabilities', 'StockholdersEquity',
    'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
    'MinorityInterest', 'LiabilitiesAndStockholdersEquity',
    'TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests',
    'TemporaryEquityCarryingAmountAttributableToParent',
})


def _day(value):
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError('exact ISO source date required')
    return result


def extract_facts(path, expected_sha256, accession, cik, report_end, filed):
    """Read the fixed default money scope; source review never enables features."""
    return _extract(path, expected_sha256, accession, cik, report_end, filed, MONEY_CONCEPTS)


def extract_balance_facts(path, expected_sha256, accession, cik, report_end, filed):
    """Read a separate accounting-audit scope without changing default features."""
    result = _extract(path, expected_sha256, accession, cik, report_end, filed, BALANCE_CONCEPTS)
    result['schema'] = 'sec-original-balance-audit-v1'
    return result


def _extract(path, expected_sha256, accession, cik, report_end, filed, selected_concepts):
    """Keep exact units/periods and every matching fact; do not choose concepts.

    This deliberately does not load schemas, linkbases, URLs or external entities.
    Only a small standard US-GAAP money-concept family is supported. Financial
    feature assembly and publication/identity verification are separate gates.
    """
    if not re.fullmatch(r'\d{10}-\d{2}-\d{6}', accession):
        raise ValueError('valid accession required')
    if not isinstance(cik, str) or not re.fullmatch(r'[0-9]{1,10}', cik) or int(cik) <= 0:
        raise ValueError('valid issuer CIK required')
    cik = str(int(cik))
    end, filing = _day(report_end), _day(filed)
    if end > filing or filing >= date(2024, 1, 1):
        raise ValueError('invalid or retired filing/report dates')
    with Path(path).open('rb') as stream:
        raw = stream.read(20_000_001)
    if len(raw) > 20_000_000:
        raise ValueError('XBRL source exceeds byte limit')
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('XBRL source checksum mismatch')
    text = raw.decode('utf-8-sig')
    if '<!DOCTYPE' in text.upper() or '<!ENTITY' in text.upper():
        raise ValueError('DTD/entity declarations are unsupported')
    namespaces = {}
    for _, item in ET.iterparse(io.StringIO(text), events=('start-ns',)):
        prefix, uri = item
        if prefix in namespaces and namespaces[prefix] != uri:
            raise ValueError('namespace rebinding unsupported')
        namespaces[prefix] = uri
    root = ET.fromstring(text)
    if root.tag != X + 'xbrl':
        raise ValueError('original extracted XBRL instance required')

    def qname(value):
        prefix, sep, name = value.partition(':')
        if not sep:
            prefix, name = '', prefix
        if prefix not in namespaces or not name:
            raise ValueError('unresolved unit QName')
        return '{' + namespaces[prefix] + '}' + name

    def indexed(tag):
        result = {}
        for node in root.findall(X + tag):
            identity = node.get('id')
            if not identity or identity in result:
                raise ValueError('unique source context/unit ID required')
            result[identity] = node
        return result

    contexts, units = indexed('context'), indexed('unit')
    declared_ends = {n.text for n in root if isinstance(n.tag, str)
                     and re.fullmatch(r'\{https?://(?:xbrl.sec.gov|xbrl.us)/dei/[^}]+\}DocumentPeriodEndDate', n.tag)}
    if declared_ends != {report_end}:
        raise ValueError('original document reporting period differs from supplied period')
    facts, counts = [], Counter()
    for node in root:
        if not re.fullmatch(r'\{https?://(?:fasb.org|xbrl.us)/us-gaap/[^}]+\}[^}]+', node.tag):
            continue
        concept = node.tag.split('}', 1)[1]
        if concept not in selected_concepts:
            continue
        counts['selected_concept_rows'] += 1
        context_id, unit_id = node.get('contextRef'), node.get('unitRef')
        if context_id not in contexts or unit_id not in units:
            raise ValueError('orphan selected fact context/unit reference')
        context, unit = contexts[context_id], units[unit_id]
        identifiers = context.findall(X + 'entity/' + X + 'identifier')
        if len(identifiers) != 1:
            raise ValueError('one context issuer required')
        identifier = identifiers[0]
        value = identifier.text or ''
        issuer_valid = (identifier.get('scheme') in {'http://www.sec.gov/CIK', 'https://www.sec.gov/CIK'}
                        and re.fullmatch(r'[0-9]{1,10}', value) and int(value) > 0)
        reasons = []
        if not issuer_valid or str(int(value)) != cik:
            reasons.append('issuer_mismatch')
        periods = context.findall(X + 'period')
        if len(periods) != 1:
            raise ValueError('one context period required')
        period = periods[0]
        instant = period.findtext(X + 'instant')
        start, finish = period.findtext(X + 'startDate'), period.findtext(X + 'endDate')
        if instant is not None and start is None and finish is None and len(period) == 1:
            kind, finish = 'instant', instant
        elif instant is None and start is not None and finish is not None and len(period) == 2:
            kind = 'duration'
            if _day(start) > _day(finish):
                raise ValueError('reversed original period')
        else:
            raise ValueError('unsupported selected financial period')
        if _day(finish) > end:
            reasons.append('period_after_report')
        segment, scenario = context.find(X + 'entity/' + X + 'segment'), context.find(X + 'scenario')
        qualified = any(n is not None and (len(n) or (n.text or '').strip()) for n in (segment, scenario))
        if qualified:
            reasons.append('segment_or_scenario')
        measures = unit.findall(X + 'measure')
        unit_measures = [qname(n.text or '') for n in measures]
        if len(unit) != 1 or unit_measures != ['{http://www.xbrl.org/2003/iso4217}USD']:
            reasons.append('not_simple_usd')
        nil = node.get(NIL, 'false')
        if nil not in {'true', 'false', '1', '0'}:
            raise ValueError('invalid XBRL nil flag')
        numeric = None
        if nil in {'true', '1'}:
            reasons.append('nil')
        elif not reasons:
            try:
                numeric = Decimal((node.text or '').strip())
                if not numeric.is_finite():
                    raise ValueError('nonfinite selected source fact')
            except InvalidOperation as error:
                raise ValueError('invalid selected source number') from error
        if reasons:
            counts['excluded_rows'] += 1
            for reason in reasons:
                counts['excluded_' + reason] += 1
        else:
            counts['unqualified_usd_rows'] += 1
        facts.append(dict(concept=node.tag, accession=accession, context_id=context_id,
                          unit_id=unit_id, unit_measures=unit_measures,
                          period_kind=kind, start=start, end=finish, cik=cik,
                          value=str(numeric) if numeric is not None else None,
                          decimals=node.get('decimals'), exclusion_reasons=reasons))
    return dict(schema='sec-original-money-facts-v1', source_sha256=expected_sha256,
                accession=accession, cik=cik, report_end=report_end, filed=filed,
                counts=dict(counts), facts=facts, financial_features_enabled=False,
                historical_mapping_verified=False, publication_verified=False,
                causal_ready=False, training_performed=False)

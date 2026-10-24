from copy import deepcopy

import pytest

from rl_stocks.features.fundamentals.balance import EQUITY, TEMPORARY, inspect_balance


def payload(temporary=None):
    rows=[]
    values={'Assets':100,'Liabilities':60,EQUITY:30,'StockholdersEquity':25}
    if temporary is not None:
        values[TEMPORARY]=temporary
    for concept,value in values.items():
        rows.append(dict(concept='{http://fasb.org/us-gaap/2019-01-31}'+concept,
                         value=str(value),period_kind='instant',end='2019-12-31',
                         accession='one',cik='1',unit_measures=['{http://www.xbrl.org/2003/iso4217}USD'],
                         exclusion_reasons=[]))
    return dict(schema='sec-original-balance-audit-v1',facts=rows,accession='one',cik='1',
                report_end='2019-12-31',source_sha256='a'*64)


def test_temporary_equity_can_explain_gap_without_becoming_liabilities():
    r=inspect_balance(payload(10))
    assert r['assets_minus_liabilities_minus_equity_including_nci']=='10'
    assert not r['three_term_reported_identity_holds']
    assert r['four_term_reported_identity_holds']
    assert r['assets_minus_liabilities_minus_parent_equity']=='15'
    assert not r['liability_inference_permitted']


def test_missing_temporary_equity_remains_unknown_not_zero():
    r=inspect_balance(payload())
    assert r['reported_total_temporary_equity'] is None
    assert not r['missing_temporary_equity_is_zero']
    assert 'four_term_reported_identity_holds' not in r


def test_missing_direct_liabilities_cannot_be_imputed_from_equity():
    p=payload(10)
    p['facts']=[r for r in p['facts'] if not r['concept'].endswith('}Liabilities')]
    r=inspect_balance(p)
    assert r['reason']=='missing_direct_assets_or_liabilities_no_inference'
    assert not r['liability_inference_permitted']


def test_excluded_or_different_period_facts_cannot_change_reconciliation():
    p=payload(10)
    extra=deepcopy(p['facts'][0])
    extra.update(value='NaN',exclusion_reasons=['segment_or_scenario'])
    p['facts'].append(extra)
    assert inspect_balance(p)['four_term_reported_identity_holds']


def test_conflicting_duplicate_values_are_not_selected():
    p=payload(10)
    other=deepcopy(p['facts'][0])
    other['value']='101'
    p['facts'].append(other)
    r=inspect_balance(p)
    assert not r['arithmetic_scope_ready'] and r['conflicting_concepts']==['Assets']


def test_mixed_issuer_or_currency_rejected():
    p=payload(10)
    p['facts'][0]['cik']='2'
    with pytest.raises(ValueError,match='coherent'):
        inspect_balance(p)


def test_low_caller_precision_cannot_create_false_balance_identity():
    from decimal import localcontext
    p=payload()
    values={'Assets':'100000000000000000000000000001','Liabilities':'2',
            EQUITY:'100000000000000000000000000000'}
    for row in p['facts']:
        key=row['concept'].split('}')[1]
        if key in values:
            row['value']=values[key]
    with localcontext() as context:
        context.prec=4
        r=inspect_balance(p)
    assert r['assets_minus_liabilities_minus_equity_including_nci']=='-1'
    assert not r['three_term_reported_identity_holds']

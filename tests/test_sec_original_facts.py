import hashlib

import pytest

from rl_stocks.data.providers.sec.facts import extract_facts


BASE = '''<xbrl xmlns="http://www.xbrl.org/2003/instance"
 xmlns:us-gaap="http://fasb.org/us-gaap/2019-01-31"
 xmlns:dei="http://xbrl.sec.gov/dei/2019-01-31"
 xmlns:iso4217="http://www.xbrl.org/2003/iso4217"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
 <context id="c"><entity><identifier scheme="http://www.sec.gov/CIK">000123</identifier>{segment}</entity>
 <period><startDate>2019-02-03</startDate><endDate>{end}</endDate></period>{scenario}</context>
 <unit id="u"><measure>iso4217:{currency}</measure></unit>
 <dei:DocumentPeriodEndDate contextRef="c">2020-02-01</dei:DocumentPeriodEndDate>
 <us-gaap:Revenues contextRef="{context}" unitRef="{unit}" decimals="-6" {nil}>{value}</us-gaap:Revenues>
 {extra}</xbrl>'''


def write(tmp_path, **changes):
    values=dict(segment='',scenario='',end='2020-02-01',currency='USD',context='c',
                unit='u',nil='',value='-12345678901234567890.1234',extra='')
    values.update(changes)
    raw=BASE.format(**values).encode()
    path=tmp_path/'instance.xml'
    path.write_bytes(raw)
    return path,hashlib.sha256(raw).hexdigest()


def extract(path, sha, **kwargs):
    args=dict(accession='0000000123-20-000001',cik='123',report_end='2020-02-01',filed='2020-04-01')
    args.update(kwargs)
    return extract_facts(path,sha,**args)


def test_exact_original_duration_and_decimal_preserved_without_feature_readiness(tmp_path):
    result=extract(*write(tmp_path))
    fact=result['facts'][0]
    assert fact['start']=='2019-02-03' and fact['end']=='2020-02-01'
    assert fact['value']=='-12345678901234567890.1234'
    assert fact['decimals']=='-6'
    assert not fact['exclusion_reasons']
    assert not result['causal_ready'] and not result['financial_features_enabled']


@pytest.mark.parametrize('changes,reason', [
    ({'end':'2024-12-31','value':'NaN'},'period_after_report'),
    ({'segment':'<segment><member>Region</member></segment>','value':'NaN'},'segment_or_scenario'),
    ({'scenario':'<scenario><forecast>Yes</forecast></scenario>','value':'NaN'},'segment_or_scenario'),
    ({'currency':'EUR'},'not_simple_usd'),
    ({'nil':'xsi:nil="true"','value':''},'nil'),
])
def test_excluded_amounts_never_enter_extraction(tmp_path,changes,reason):
    result=extract(*write(tmp_path,**changes))
    assert reason in result['facts'][0]['exclusion_reasons']
    assert result['facts'][0]['value'] is None


@pytest.mark.parametrize('changes', [{'context':'orphan'},{'unit':'orphan'},
                                    {'value':'NaN'},{'value':'INF'}, {'end':'2020-02-30'}])
def test_invalid_eligible_facts_and_references_rejected(tmp_path,changes):
    with pytest.raises(ValueError):
        extract(*write(tmp_path,**changes))


def test_archive_month_end_cannot_replace_original_period(tmp_path):
    with pytest.raises(ValueError,match='reporting period differs'):
        extract(*write(tmp_path),report_end='2020-01-31')


def test_retired_filing_and_source_tampering_rejected(tmp_path):
    path,sha=write(tmp_path)
    with pytest.raises(ValueError,match='retired'):
        extract(path,sha,filed='2024-01-01')
    with pytest.raises(ValueError,match='checksum'):
        extract(path,'0'*64)


def test_dtd_rejected_before_parsing(tmp_path):
    path,sha=write(tmp_path)
    raw=b'<!DOCTYPE xbrl [<!ENTITY x "bad">]>'+path.read_bytes()
    path.write_bytes(raw)
    with pytest.raises(ValueError,match='DTD'):
        extract(path,hashlib.sha256(raw).hexdigest())


def test_duplicate_facts_retained_without_silent_concept_selection(tmp_path):
    extra='<us-gaap:Revenues contextRef="c" unitRef="u">999</us-gaap:Revenues>'
    result=extract(*write(tmp_path,extra=extra))
    assert len(result['facts'])==2
    assert result['facts'][0]['value']!=result['facts'][1]['value']
    assert not result['causal_ready']


def test_legacy_2009_us_gaap_and_dei_namespaces_supported_explicitly(tmp_path):
    path,sha=write(tmp_path)
    raw=path.read_bytes().replace(b'http://fasb.org/us-gaap/2019-01-31',b'http://xbrl.us/us-gaap/2009-01-31').replace(b'http://xbrl.sec.gov/dei/2019-01-31',b'http://xbrl.us/dei/2009-01-31')
    path.write_bytes(raw)
    result=extract(path,hashlib.sha256(raw).hexdigest())
    assert result['counts']['unqualified_usd_rows']==1
    assert result['facts'][0]['concept'].startswith('{http://xbrl.us/us-gaap/2009-01-31}')


def test_balance_scope_is_separate_and_does_not_expand_default_money_features(tmp_path):
    from rl_stocks.data.providers.sec.facts import extract_balance_facts
    extra='''<context id="instant"><entity><identifier scheme="http://www.sec.gov/CIK">123</identifier></entity><period><instant>2020-02-01</instant></period></context>
    <us-gaap:Assets contextRef="instant" unitRef="u">100</us-gaap:Assets>
    <us-gaap:StockholdersEquity contextRef="instant" unitRef="u">30</us-gaap:StockholdersEquity>'''
    path,sha=write(tmp_path,extra=extra)
    default=extract(path,sha)
    audit=extract_balance_facts(path,sha,'0000000123-20-000001','123','2020-02-01','2020-04-01')
    assert audit['schema']=='sec-original-balance-audit-v1'
    assert any(r['concept'].endswith('}StockholdersEquity') for r in audit['facts'])
    assert not any(r['concept'].endswith('}StockholdersEquity') for r in default['facts'])
    assert not any(r['concept'].endswith('}Revenues') for r in audit['facts'])

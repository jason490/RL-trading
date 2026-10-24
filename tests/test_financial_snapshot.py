from copy import deepcopy

import pytest

from rl_stocks.features.fundamentals.snapshot import build_snapshot

NS = '{http://fasb.org/us-gaap/2019-01-31}'


def fact(concept,value,start=None,end='2020-02-01',**extra):
    return dict(concept=NS+concept,value=str(value),start=start,end=end,
                period_kind='duration' if start else 'instant',accession='source',cik='1',
                unit_measures=['{http://www.xbrl.org/2003/iso4217}USD'],
                exclusion_reasons=[],**extra)


def payload():
    return dict(schema='sec-original-money-facts-v1',report_end='2020-02-01',filed='2020-04-01',
                source_sha256='a'*64,accession='source',cik='1',facts=[
                    fact('Revenues',120,'2019-02-03'),fact('NetIncomeLoss',-12,'2019-02-03'),
                    fact('Revenues',100,'2018-02-04','2019-02-02'),
                    fact('Assets',100),fact('Liabilities',60)])


def test_reported_52_week_ratios_negative_margin_and_immutable_source():
    p=payload()
    before=deepcopy(p)
    result=build_snapshot(p,'10-K')
    assert result['arithmetic_ready']
    assert result['values']==pytest.approx([0.2,-0.1,0.6])
    assert result['interval_days']==364
    assert not result['financial_features_enabled']
    assert not result['publication_verified']
    assert p==before


def test_annual_uses_long_interval_not_quarter_detail():
    p=payload()
    p['facts'] += [fact('Revenues',999,'2019-11-10'),fact('NetIncomeLoss',100,'2019-11-10')]
    assert build_snapshot(p,'10-K')['values']==pytest.approx([0.2,-0.1,0.6])


def test_sixteen_week_quarter_is_not_assumed_ninety_days():
    p=payload()
    p.update(report_end='2019-05-25',filed='2019-06-05')
    p['facts']=[fact('Revenues',120,'2019-02-03','2019-05-25'),
                fact('NetIncomeLoss',-12,'2019-02-03','2019-05-25'),
                fact('Revenues',100,'2018-02-04','2018-05-26'),
                fact('Assets',100,end='2019-05-25'),fact('Liabilities',60,end='2019-05-25')]
    result=build_snapshot(p,'10-Q')
    assert result['arithmetic_ready'] and result['interval_days']==112


def test_agreeing_duplicates_allowed_conflicting_duplicates_never_win():
    p=payload()
    p['facts'].append(fact('Revenues','120.0000','2019-02-03'))
    assert build_snapshot(p,'10-K')['arithmetic_ready']
    p['facts'].append(fact('Revenues',130,'2019-02-03'))
    assert build_snapshot(p,'10-K')['reason']=='conflicting_current_flow_values'
    p['facts'].reverse()
    assert not build_snapshot(p,'10-K')['arithmetic_ready']


def test_disagreeing_total_concepts_not_selected_by_priority_or_summed():
    p=payload()
    p['facts'].append(fact('RevenueFromContractWithCustomerExcludingAssessedTax',130,'2019-02-03'))
    assert build_snapshot(p,'10-K')['reason']=='conflicting_current_flow_values'


def test_ambiguous_prior_intervals_rejected_not_nearest_selected():
    p=payload()
    p['facts'].append(fact('Revenues',100,'2018-02-05','2019-02-03'))
    assert build_snapshot(p,'10-K')['reason']=='missing_or_ambiguous_prior_interval'


def test_revenue_concept_change_without_common_comparable_rejected():
    p=payload()
    p['facts'][2]['concept']=NS+'SalesRevenueNet'
    assert build_snapshot(p,'10-K')['reason']=='no_same_revenue_concept_comparable'


@pytest.mark.parametrize('index,value',[(0,0),(2,0),(3,0),(4,-1)])
def test_invalid_denominators_or_liabilities_not_filled(index,value):
    p=payload()
    p['facts'][index]['value']=str(value)
    assert build_snapshot(p,'10-K')['reason']=='invalid_denominator_or_liabilities'


def test_excluded_amounts_cannot_change_feature_values():
    p=payload()
    bad=fact('Revenues','NaN','2019-02-03')
    bad['exclusion_reasons']=['segment_or_scenario']
    p['facts'].append(bad)
    assert build_snapshot(p,'10-K')['values']==pytest.approx([0.2,-0.1,0.6])


@pytest.mark.parametrize('key,value',[('accession','another-vintage'),('cik','2'),
                                      ('unit_measures',['EUR']),('value','NaN'),('end','2024-01-01')])
def test_inconsistent_unqualified_source_rejected(key,value):
    p=payload()
    p['facts'][0][key]=value
    with pytest.raises(ValueError):
        build_snapshot(p,'10-K')


def test_retired_filing_rejected():
    p=payload()
    p['filed']='2024-01-01'
    with pytest.raises(ValueError,match='retired'):
        build_snapshot(p,'10-K')

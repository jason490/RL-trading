from copy import deepcopy

import pytest

from rl_stocks.features.fundamentals.history import ResearchFinancialHistory
from rl_stocks.features.fundamentals.snapshot import NAMES


def entry(number=1,end='2019-12-31',filed='2020-02-25',ready=True,form='10-K',cik='1'):
    return dict(snapshot=dict(schema='filing-financial-snapshot-v1',names=list(NAMES),form=form,
        accession=f'0000000001-20-{number:06d}',cik=cik,source_sha256='a'*64,
        report_end=end,filed=filed,arithmetic_ready=ready,values=[0.1,0.2,0.3] if ready else None),
        accepted_source=filed+' 16:54:20',index_source_sha256='b'*64)


def test_source_cannot_backfill_before_publication_and_unknown_cik_not_substituted():
    h=ResearchFinancialHistory([entry()])
    assert not h.at('1','2020-02-26T21:00:00Z')['source_ready']
    after=h.at('1','2020-02-27T21:00:00Z')
    assert after['source_ready'] and not after['financial_features_enabled']
    assert not h.at('2','2020-02-27T21:00:00Z')['source_ready']


def test_future_amendment_does_not_rewrite_earlier_values_and_copies_are_independent():
    original=entry()
    amendment=entry(2,filed='2020-06-01',form='10-K/A')
    amendment['snapshot']['values']=[0.4,0.5,0.6]
    inputs=[original,amendment]
    before=deepcopy(inputs)
    h=ResearchFinancialHistory(inputs)
    assert h.at('1','2020-03-02T21:00:00Z')['values']==[0.1,0.2,0.3]
    assert h.at('1','2020-06-03T20:00:00Z')['values']==[0.4,0.5,0.6]
    result=h.at('1','2020-03-02T21:00:00Z')
    result['values'][0]=999
    original['snapshot']['values'][0]=888
    assert h.at('1','2020-03-02T21:00:00Z')['values']==[0.1,0.2,0.3]
    assert amendment==before[1]


def test_older_report_amendment_cannot_displace_newer_period():
    older=entry(2,end='2018-12-31',filed='2020-06-01',form='10-K/A')
    h=ResearchFinancialHistory([entry(),older])
    assert h.at('1','2020-06-03T20:00:00Z')['accession'].endswith('000001')


def test_new_unready_report_blocks_fallback_to_old_ready_snapshot():
    newer=entry(2,end='2020-03-31',filed='2020-04-28',ready=False,form='10-Q')
    h=ResearchFinancialHistory([entry(),newer])
    result=h.at('1','2020-04-30T20:00:00Z')
    assert not result['source_ready'] and result['values'] is None
    assert result['reason']=='latest_snapshot_arithmetic_unready'


def test_amendment_does_not_reset_reporting_period_staleness():
    amendment=entry(2,filed='2021-04-01',form='10-K/A')
    h=ResearchFinancialHistory([entry(),amendment])
    result=h.at('1','2021-04-05T20:00:00Z')
    assert result['reason']=='stale_reporting_period'
    assert result['values'] is None


def test_conflicting_equal_ranked_accessions_reject_instead_of_tie_break():
    h=ResearchFinancialHistory([entry(),entry(2)])
    with pytest.raises(ValueError,match='ambiguous'):
        h.at('1','2020-03-02T21:00:00Z')


def test_naive_decisions_retired_sources_and_retired_queries_rejected():
    h=ResearchFinancialHistory([entry()])
    with pytest.raises(ValueError,match='aware'):
        h.at('1','2020-03-02T21:00:00')
    with pytest.raises(ValueError,match='retired'):
        h.at('1','2024-01-02T21:00:00Z')
    with pytest.raises(ValueError,match='retired'):
        ResearchFinancialHistory([entry(filed='2024-01-01')])


def test_invalid_ready_and_duplicate_sources_rejected():
    bad=entry()
    bad['snapshot']['values'][0]=float('nan')
    with pytest.raises(ValueError,match='finite'):
        ResearchFinancialHistory([bad])
    with pytest.raises(ValueError,match='duplicate accession'):
        ResearchFinancialHistory([entry(),entry()])

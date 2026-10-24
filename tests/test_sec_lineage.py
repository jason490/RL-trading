from copy import deepcopy

import pytest

from rl_stocks.data.providers.sec.lineage import conservative_availability, issuer_asof


def record(cik='1',effective='2015-01-02',available='2015-01-06T14:30:00Z',**extra):
    return dict(symbol='ABC',cik=cik,effective_session=effective,available_time=available,
                kind='observation',source_sha256='a'*64,
                source_url='https://www.sec.gov/Archives/example.htm',**extra)


def test_publication_and_effective_dates_both_gate_without_backfill():
    event=record(cik='2',effective='2015-10-02',available='2015-10-06T13:30:00Z')
    event.update(kind='successor',predecessor_cik='1')
    assert issuer_asof([event],'ABC','2010-01-04','2010-01-04T21:00:00Z') is None
    assert issuer_asof([event],'ABC','2015-10-05','2015-10-05T20:00:00Z') is None
    result=issuer_asof([event],'ABC','2015-10-06','2015-10-06T20:00:00Z')
    assert result['cik']=='2'
    assert not result['financial_features_enabled']
    assert not result['historical_price_link_verified']


def test_unavailable_future_event_cannot_change_earlier_visible_issuer():
    old=record()
    new=record(cik='2',effective='2015-10-02',available='2015-10-06T13:30:00Z')
    new.update(kind='successor',predecessor_cik='1')
    original=deepcopy([old,new])
    assert issuer_asof([old,new],'ABC','2015-10-05','2015-10-05T20:00:00Z')['cik']=='1'
    assert issuer_asof([old,new],'ABC','2015-10-06','2015-10-06T20:00:00Z')['cik']=='2'
    assert [old,new]==original


def test_ordinary_subsidiary_observation_cannot_replace_public_parent():
    old=record()
    unexpected=record(cik='2',effective='2015-03-01',available='2015-03-03T14:30:00Z')
    with pytest.raises(ValueError,match='transition'):
        issuer_asof([old,unexpected],'ABC','2015-03-04','2015-03-04T21:00:00Z')


def test_announced_future_transition_not_effective_yet():
    old=record()
    new=record(cik='2',effective='2015-10-02',available='2015-08-12T13:30:00Z')
    new.update(kind='successor',predecessor_cik='1')
    assert issuer_asof([old,new],'ABC','2015-09-01','2015-09-01T20:00:00Z')['cik']=='1'


@pytest.mark.parametrize('filed,accepted,expected', [
    ('2015-10-02','2015-10-02 16:30:00','2015-10-06T13:30:00Z'),
    ('2020-02-28','2020-02-27 17:44:00.0','2020-03-03T14:30:00Z'),
    ('2020-07-02','2020-07-02 16:30:00','2020-07-07T13:30:00Z'),
])
def test_fixed_lag_handles_weekends_dst_and_holiday(filed,accepted,expected):
    result=conservative_availability(filed,accepted)
    assert result['available_time']==expected
    assert result['availability_reconstructed']
    assert not result['exact_first_publication_verified']


def test_retired_source_and_naive_decision_rejected():
    with pytest.raises(ValueError,match='retired'):
        conservative_availability('2024-01-01','2024-01-01 10:00:00')
    with pytest.raises(ValueError,match='aware'):
        issuer_asof([record()],'ABC','2015-01-07','2015-01-07T21:00:00')


def test_late_2023_source_cannot_become_available_in_retired_year_early():
    result=conservative_availability('2023-12-29','2023-12-29 16:30:00')
    assert result['available_time']=='2024-01-03T14:30:00Z'

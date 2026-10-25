"""Offline official annual XML contracts and retired-year rejection."""

import pytest

from rl_stocks.data.providers.treasury.xml import annual_url, parse_annual


def document(entries, extra=""):
    return (f'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata" xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices">{extra}' + ''.join('<entry><content><m:properties>'+row+'</m:properties></content></entry>' for row in entries) + '</feed>').encode()


def row(day="2020-01-02", two="1.58", ten="1.88"):
    return f'<d:NEW_DATE>{day}T00:00:00</d:NEW_DATE><d:BC_2YEAR>{two}</d:BC_2YEAR><d:BC_10YEAR>{ten}</d:BC_10YEAR>'


def test_preserves_percent_decimal_units_orders_dates_and_accepts_negative():
    parsed = parse_annual(document([row('2020-01-03', '-0.01', '1.880'), row()]), 2020)
    assert parsed == [dict(observation_date='2020-01-02', BC_2YEAR='1.58', BC_10YEAR='1.88'), dict(observation_date='2020-01-03', BC_2YEAR='-0.01', BC_10YEAR='1.880')]
    assert annual_url(2020).endswith('data=daily_treasury_yield_curve&field_tdr_date_value=2020')


@pytest.mark.parametrize('year', [2024, 2026, 2008, True, '2020'])
def test_no_retired_or_unbounded_request(year):
    with pytest.raises(ValueError):
        annual_url(year)


@pytest.mark.parametrize('bad', [row('2024-01-02'), row('2019-12-31'), row('2020-01-04'), row(two='NaN'), row(two='inf'), row(two='101'), row(two='not numeric')])
def test_source_scope_and_nonfinite_or_invalid_facts_reject(bad):
    with pytest.raises(ValueError):
        parse_annual(document([bad]), 2020)


def test_missing_yields_retained_not_zero_filled():
    missing = '<d:NEW_DATE>2020-01-03T00:00:00</d:NEW_DATE><d:BC_2YEAR m:null="true"/>'
    assert parse_annual(document([row(), missing]), 2020)[1] == dict(observation_date='2020-01-03', BC_2YEAR=None, BC_10YEAR=None)
    with pytest.raises(ValueError):
        parse_annual(document([missing]), 2020)


@pytest.mark.parametrize('raw', [document([row(), row()]), document([row() + '<d:BC_2YEAR>1</d:BC_2YEAR>']), document([row()], '<link rel="next" href="?field_tdr_date_value=all"/>'), b'<!DOCTYPE feed [<!ENTITY x "bad">]><feed/>', b'<html>denied</html>', b'x'*(2*1024*1024+1), document([])])
def test_duplicates_pagination_entities_html_and_size_reject(raw):
    with pytest.raises(ValueError):
        parse_annual(raw, 2020)



def test_non_utf8_declaration_cannot_bypass_entity_rejection():
    raw = '<?xml version="1.0" encoding="utf-16"?><!DOCTYPE feed [<!ENTITY x "bad">]><feed>&x;</feed>'.encode('utf-16')
    with pytest.raises(ValueError, match='UTF8'):
        parse_annual(raw, 2020)

from hashlib import sha256

import pytest

from rl_stocks.data.providers.sec.html_stock_type import inspect_html_stock


def inspect(tmp_path, text, symbol='ABT'):
    raw = text.encode()
    path = tmp_path / 'primary.htm'
    path.write_bytes(raw)
    return inspect_html_stock(path, sha256(raw).hexdigest(), symbol=symbol)


def test_explicit_market_clause_is_only_an_unadmitted_review_hint(tmp_path):
    result = inspect(tmp_path, '<html>Securities registered Common Stock NYSE. '
                     'Our common stock trades on NYSE under the symbol “ABT”.</html>')
    assert result['common_equity_cover_hint'] and result['explicit_ticker_clause_hint']
    assert not result['historical_metadata_admitted']
    assert len(result['first_ticker_clause']['review_excerpt'].split()) <= 18


def test_drug_code_company_abbreviation_and_embedded_scripts_do_not_match(tmp_path):
    text = '<html>Securities registered Common Shares. ABT-874 is a drug. ABT common stock. '
    text += '<script>common stock under symbol ABT</script></html>'
    result = inspect(tmp_path, text)
    assert result['common_equity_cover_hint'] and not result['explicit_ticker_clause_hint']
    for wrong in ('ABTC', 'ABT-874', 'ABT.A'):
        assert not inspect(tmp_path, f'<html>Our common stock under symbol {wrong}.</html>')[
            'explicit_ticker_clause_hint']


def test_checksum_and_noncanonical_symbol_reject(tmp_path):
    path = tmp_path / 'primary.htm'
    path.write_text('<html/>')
    with pytest.raises(ValueError, match='checksum'):
        inspect_html_stock(path, 'a' * 64, symbol='ABT')
    with pytest.raises(ValueError, match='ticker'):
        inspect(tmp_path, '<html/>', symbol='not-a-symbol')

from hashlib import sha256

import pytest

from rl_stocks.data.providers.sec.instrument_type import inspect_stock_type


def source(symbol='AAPL', *, issuer='320193', namespace='http://xbrl.us/dei/2009-01-31',
           segment='', shares='1', form='10-K'):
    return f'''<xbrl xmlns="http://www.xbrl.org/2003/instance"
      xmlns:x="http://www.xbrl.org/2003/instance" xmlns:dei="{namespace}">
      <context id="c"><entity><identifier scheme="http://www.sec.gov/CIK">{issuer}</identifier>
      {segment}</entity><period><instant>2009-10-16</instant></period></context>
      <unit id="s"><measure>x:shares</measure></unit>
      <dei:TradingSymbol contextRef="c">{symbol}</dei:TradingSymbol>
      <dei:EntityRegistrantName contextRef="c">APPLE INC</dei:EntityRegistrantName>
      <dei:DocumentType contextRef="c">{form}</dei:DocumentType>
      <dei:EntityCommonStockSharesOutstanding contextRef="c" unitRef="s">{shares}</dei:EntityCommonStockSharesOutstanding>
      </xbrl>'''.encode()


def inspect(tmp_path, raw):
    path = tmp_path / 'instance.xml'
    path.write_bytes(raw)
    return inspect_stock_type(path, sha256(raw).hexdigest(), cik='320193', symbol='AAPL',
                              form='10-K', filed='2009-10-27')


def test_exact_document_hints_do_not_enable_historical_input(tmp_path):
    result = inspect(tmp_path, source())
    assert result['document_exact_stock_identity_hint']
    assert not result['historical_metadata_admitted']
    assert not result['validity_interval_inferred']
    assert not result['model_inputs_enabled']
    shares = [f for f in result['facts'] if f['concept'] == 'EntityCommonStockSharesOutstanding']
    assert shares and 'text' not in shares[0]


@pytest.mark.parametrize('change', [dict(symbol='OTHER'), dict(issuer='1234'),
                                    dict(namespace='http://example.com/dei/2009'),
                                    dict(segment='<segment/>'), dict(shares='NaN'),
                                    dict(form='10-Q')])
def test_wrong_issuer_symbol_namespace_or_qualified_fact_is_not_evidence(tmp_path, change):
    assert not inspect(tmp_path, source(**change))['document_exact_stock_identity_hint']


def test_checksum_and_external_declarations_reject(tmp_path):
    path = tmp_path / 'instance.xml'
    path.write_bytes(source())
    with pytest.raises(ValueError, match='checksum'):
        inspect_stock_type(path, 'a' * 64, cik='320193', symbol='AAPL', form='10-K',
                           filed='2009-10-27')
    with pytest.raises(ValueError, match='DTD'):
        inspect(tmp_path, b'<!DOCTYPE xbrl []>' + source())

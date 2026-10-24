import hashlib

import pytest

from rl_stocks.data.providers.sec.identity import inspect_identity


def source(tmp_path, extra=''):
    raw=('''<html><ix:nonNumeric name="dei:EntityCentralIndexKey" contextRef="issuer">000123</ix:nonNumeric>
    <ix:nonNumeric name="dei:TradingSymbol" contextRef="common"><span>ABC</span></ix:nonNumeric>
    <ix:nonNumeric name="dei:TradingSymbol" contextRef="bond">ABC25</ix:nonNumeric>
    <ix:nonNumeric name="dei:Security12bTitle" contextRef="common">Common&#160;Stock</ix:nonNumeric>
    '''+extra+'</html>').encode()
    path=tmp_path/'filing.htm'
    path.write_bytes(raw)
    return path,hashlib.sha256(raw).hexdigest()


def test_match_preserves_alternate_securities_without_backdating(tmp_path):
    path,sha=source(tmp_path)
    result=inspect_identity(path,sha,'123','ABC')
    assert result['issuer_and_ticker_match']
    assert result['declared_symbols']==['ABC','ABC25']
    assert any(r['value']=='Common Stock' for r in result['facts'])
    assert not result['historical_mapping_verified']
    assert not result['validity_interval_inferred']
    assert not result['model_inputs_enabled']


def test_expected_current_parent_or_substring_cannot_match(tmp_path):
    path,sha=source(tmp_path)
    assert not inspect_identity(path,sha,'456','ABC')['issuer_and_ticker_match']
    assert not inspect_identity(path,sha,'123','AB')['issuer_and_ticker_match']


def test_multiple_issuers_remain_ambiguous(tmp_path):
    path,sha=source(tmp_path, '<ix:nonNumeric name="dei:EntityCentralIndexKey">456</ix:nonNumeric>')
    assert not inspect_identity(path,sha,'123','ABC')['issuer_and_ticker_match']


def test_corrupt_source_and_incomplete_facts_rejected(tmp_path):
    path,sha=source(tmp_path)
    with pytest.raises(ValueError,match='checksum'):
        inspect_identity(path,'0'*64,'123','ABC')
    path.write_bytes(b'<ix:nonNumeric name="dei:TradingSymbol">ABC')
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='unterminated'):
        inspect_identity(path,sha,'123','ABC')


def test_malformed_additional_issuer_does_not_disappear_into_positive_match(tmp_path):
    path,sha=source(tmp_path, '<ix:nonNumeric name="dei:EntityCentralIndexKey">unknown</ix:nonNumeric>')
    assert not inspect_identity(path,sha,'123','ABC')['issuer_and_ticker_match']

import io
import json
from urllib.error import HTTPError

import pytest

from rl_stocks.data.providers.sec import download as d


class Response(io.BytesIO):
    status = 200

    def __init__(self, body, declared=None):
        super().__init__(body)
        self.headers = {} if declared is None else {'Content-Length': declared}


class Opener:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        result = next(self.actions)
        if isinstance(result, Exception):
            raise result
        return result


def setup(monkeypatch, actions):
    opener = Opener(actions)
    monkeypatch.setattr(d, 'build_opener', lambda *args: opener)
    monkeypatch.setattr(d.time, 'sleep', lambda value: None)
    return opener


def test_acquisition_preserves_bytes_and_excludes_identity_from_receipt(tmp_path, monkeypatch):
    body = b'PK\x03\x04fixture'
    opener = setup(monkeypatch, [Response(body, str(len(body)))])
    path = d.download_archive(tmp_path, '2020q1', 'Research contact@example.org')
    assert path.read_bytes() == body
    receipts = list(path.parent.glob('acquisition-*.json'))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt['archive_sha256'] == path.parent.name
    assert not receipt['causal_ready']
    assert 'contact@example.org' not in receipts[0].read_text()
    assert opener.requests[0].get_header('User-agent') == 'Research contact@example.org'
    assert not list(tmp_path.glob('tmp*'))


@pytest.mark.parametrize('code', [403, 429, 404])
def test_access_rejection_never_retries(tmp_path, monkeypatch, code):
    opener = setup(monkeypatch, [HTTPError('url', code, 'private detail', {}, None)])
    with pytest.raises(ValueError, match=f'HTTP {code}'):
        d.download_archive(tmp_path, '2020q1', 'Research contact@example.org')
    assert len(opener.requests) == 1
    assert not list(tmp_path.glob('*/archive.zip'))


def test_transient_retries_bounded(tmp_path, monkeypatch):
    opener = setup(monkeypatch, [HTTPError('url', 503, 'busy', {}, None) for _ in range(3)])
    with pytest.raises(ValueError, match='503'):
        d.download_archive(tmp_path, '2020q1', 'Research contact@example.org')
    assert len(opener.requests) == 3


@pytest.mark.parametrize('body,length,limit', [
    (b'<html>blocked</html>', None, 100),
    (b'PK\x03\x04too large', None, 5),
    (b'PK\x03\x04data', '999', 10),
    (b'PK\x03\x04data', '9', 100),
])
def test_bad_downloads_leave_no_archive_or_temporary(tmp_path, monkeypatch, body, length, limit):
    setup(monkeypatch, [Response(body, length)])
    with pytest.raises(ValueError):
        d.download_archive(tmp_path, '2020q1', 'Research contact@example.org', max_bytes=limit)
    assert not list(tmp_path.glob('*/archive.zip'))
    assert not list(tmp_path.glob('tmp*'))


def test_identity_is_private_and_dotenv_not_executed(tmp_path, monkeypatch):
    monkeypatch.delenv('SEC_USER_AGENT', raising=False)
    config = tmp_path / '.env'
    config.write_text('SEC_USER_AGENT="Research contact@example.org"\nOTHER=$(false)\n')
    assert d.identity(config) == 'Research contact@example.org'
    monkeypatch.setenv('SEC_USER_AGENT', 'Research env@example.org')
    assert d.identity(config) == 'Research env@example.org'
    monkeypatch.delenv('SEC_USER_AGENT')
    config.write_text('OTHER=ignored\n')
    with pytest.raises(ValueError, match='Configure SEC_USER_AGENT'):
        d.identity(config)


def test_scope_and_identity_fail_before_request(tmp_path, monkeypatch):
    opener = setup(monkeypatch, [])
    with pytest.raises(ValueError):
        d.download_archive(tmp_path, '2024q1', 'Research contact@example.org')
    with pytest.raises(ValueError):
        d.download_archive(tmp_path, '2020q1', 'Research\r\ncontact@example.org')
    assert not opener.requests


def test_redirect_does_not_forward_identity():
    with pytest.raises(ValueError, match='redirect'):
        d._NoRedirect().redirect_request(None, None, 302, '', {}, 'https://elsewhere.invalid')


def test_cache_reuse_requires_exact_retained_bytes(tmp_path, monkeypatch):
    body = b'PK\x03\x04fixture'
    opener = setup(monkeypatch, [Response(body)])
    path = d.download_archive(tmp_path, '2020q1', 'Research contact@example.org')
    assert d.download_archive(tmp_path, '2020q1', 'Research contact@example.org') == path
    assert len(opener.requests) == 1
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='corruption'):
        d.download_archive(tmp_path, '2020q1', 'Research contact@example.org')
    assert len(opener.requests) == 1


def test_receipt_tampering_rejected_before_cache_reuse(tmp_path, monkeypatch):
    setup(monkeypatch, [Response(b'PK\x03\x04fixture')])
    path = d.download_archive(tmp_path, '2020q1', 'Research contact@example.org')
    receipt = next(path.parent.glob('acquisition-*.json'))
    receipt.write_text('{}')
    with pytest.raises(ValueError, match='receipt corruption'):
        d.download_archive(tmp_path, '2020q1', 'Research contact@example.org')


def test_master_endpoint_is_fixed_and_distinct_from_financial_archive(tmp_path,monkeypatch):
    opener=setup(monkeypatch,[Response(b'PK\x03\x04fixture')])
    d.download_master_archive(tmp_path,'2020q1','Research contact@example.org')
    assert opener.requests[0].full_url=='https://www.sec.gov/Archives/edgar/full-index/2020/QTR1/master.zip'
    with pytest.raises(ValueError):
        d.download_master_archive(tmp_path,'2024q1','Research contact@example.org')
    with pytest.raises(ValueError):
        d.download_master_archive(tmp_path,'2020q1','Research contact@example.org',max_bytes=10_000_001)


def test_cached_archive_cannot_bypass_a_smaller_request_limit(tmp_path,monkeypatch):
    setup(monkeypatch,[Response(b'PK\x03\x04fixture')])
    d.download_archive(tmp_path,'2020q1','Research contact@example.org')
    with pytest.raises(ValueError,match='cached SEC archive exceeds'):
        d.download_archive(tmp_path,'2020q1','Research contact@example.org',max_bytes=4)

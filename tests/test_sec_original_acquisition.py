import io
import json
from urllib.error import HTTPError

import pytest

from rl_stocks.data.providers.sec import originals as o


def record():
    return dict(cik='1', accession='0000000001-20-000001', instance_filename='original.xml',
                archive_filed='20200102', source_url='https://www.sec.gov/Archives/edgar/data/1/000000000120000001/original.xml')


class Response(io.BytesIO):
    status = 200

    def __init__(self, body, declared=None):
        super().__init__(body)
        self.headers = {} if declared is None else {'Content-Length': declared}


def setup(monkeypatch, action):
    calls = []
    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            if isinstance(action, Exception):
                raise action
            return action
    monkeypatch.setattr(o, 'build_opener', lambda *args: Opener())
    monkeypatch.setattr(o.time, 'sleep', lambda _: None)
    return calls


def test_original_single_request_cache_and_private_receipt(tmp_path, monkeypatch):
    body = b'<xbrl>retained source</xbrl>'
    calls = setup(monkeypatch, Response(body, str(len(body))))
    path = o.acquire_original(tmp_path, record(), 'Research private@example.org')
    assert path.read_bytes() == body
    assert o.acquire_original(tmp_path, record(), 'Research private@example.org') == path
    assert len(calls) == 1
    receipt = next((tmp_path / 'original-receipts').glob('*.json'))
    assert 'private@example.org' not in receipt.read_text()
    assert not json.loads(receipt.read_text())['causal_ready']
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='corruption'):
        o.acquire_original(tmp_path, record(), 'Research private@example.org')
    assert len(calls) == 1


@pytest.mark.parametrize('code', [403, 404, 429, 503])
def test_original_http_errors_never_retry(tmp_path, monkeypatch, code):
    calls = setup(monkeypatch, HTTPError('url', code, 'private details', {}, None))
    with pytest.raises(o.OriginalMissing if code == 404 else ValueError):
        o.acquire_original(tmp_path, record(), 'Research private@example.org')
    assert len(calls) == 1
    assert not list(tmp_path.glob('*/instance.xml'))


@pytest.mark.parametrize('body,length', [(b'', None), (b'abc', '4'), (b'abc', '20000001')])
def test_invalid_source_not_retained(tmp_path, monkeypatch, body, length):
    setup(monkeypatch, Response(body, length))
    with pytest.raises(ValueError):
        o.acquire_original(tmp_path, record(), 'Research private@example.org')
    assert not list(tmp_path.glob('*/instance.xml'))


@pytest.mark.parametrize('key,value', [('source_url', 'https://elsewhere.invalid/file.xml'),
                                      ('instance_filename', '../original.xml'),
                                      ('archive_filed', '20240102')])
def test_endpoint_and_scope_rejected_before_network(tmp_path, monkeypatch, key, value):
    calls = setup(monkeypatch, Response(b'unused'))
    r = record()
    r[key] = value
    with pytest.raises(ValueError):
        o.acquire_original(tmp_path, r, 'Research private@example.org')
    assert not calls


def test_declared_oversized_original_is_typed_without_reading_body(tmp_path, monkeypatch):
    calls = setup(monkeypatch, Response(b'unread', '20000001'))
    with pytest.raises(o.OriginalTooLarge) as caught:
        o.acquire_original(tmp_path, record(), 'Research private@example.org')
    assert caught.value.observed_bytes == 0 and caught.value.declared_bytes == 20000001
    assert len(calls) == 1 and not list(tmp_path.glob('*/instance.xml'))


def test_streaming_oversize_reports_consumed_bytes_without_retaining_partial_source(tmp_path, monkeypatch):
    calls = setup(monkeypatch, Response(b'x' * 20_000_001))
    with pytest.raises(o.OriginalTooLarge) as caught:
        o.acquire_original(tmp_path, record(), 'Research private@example.org')
    assert caught.value.observed_bytes == 20000001
    assert len(calls) == 1 and not list(tmp_path.glob('*/instance.xml'))

"""Credential isolation and request-level quota checks without network access."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rl_stocks.data.providers.alpaca.credentials import credentials
from rl_stocks.data.providers.alpaca.transport import BoundedSession, RequestPacer


def test_credentials_sources_and_secret_safe_errors():
    with TemporaryDirectory() as directory, patch.dict('os.environ', {}, clear=True):
        path = Path(directory) / '.env'
        path.write_text('OTHER=$(not_executed)\nexport APCA_API_KEY_ID="file-key"\n'
                        "APCA_API_SECRET_KEY='file-secret' # comment\n")
        assert credentials(path) == ('file-key', 'file-secret')
        with patch.dict('os.environ', {'ALPACA_API_KEY': 'env-key',
                                       'ALPACA_SECRET_KEY': 'env-secret'}):
            assert credentials(path) == ('env-key', 'env-secret')
        with patch.dict('os.environ', {'APCA_API_KEY_ID': 'partial-key'}):
            assert credentials(path) == ('file-key', 'file-secret')
        path.write_text('APCA_API_KEY_ID="sensitive-unclosed\n')
        with pytest.raises(ValueError) as error:
            credentials(path)
        assert 'sensitive' not in str(error.value)
        path.write_text('ALPACA_API_KEY=legacy\nALPACA_SECRET_KEY=secret\n')
        assert credentials(path) == ('legacy', 'secret')
        path.unlink()
        with pytest.raises(ValueError, match='Set APCA_API_KEY_ID'):
            credentials(path)


def test_request_quota_pagination_retries_and_read_only():
    now = [0.0]
    def sleep(seconds):
        now[0] += seconds
    pacer = RequestPacer(clock=lambda: now[0], sleep=sleep)
    first, second = BoundedSession(pacer), BoundedSession(pacer)
    with patch('requests.Session.request', return_value=SimpleNamespace(status_code=200)) as send:
        times = []
        for index in range(201):
            (first if index % 2 else second).get('https://data.alpaca.markets/v2/stocks/bars')
            times.append(now[0])
        assert times[180] > 60
        assert times[-1] > 67
        assert send.call_args.kwargs['timeout'] == (5, 30)
        with pytest.raises(ValueError, match='only supports GET'):
            first.post('https://data.alpaca.markets/v2/stocks/bars')
        assert send.call_count == 201
    with patch('requests.Session.request', return_value=SimpleNamespace(
            status_code=429, headers={'Retry-After': '10'})):
        first.get('https://data.alpaca.markets/v2/stocks/bars')
    limited_at = now[0]
    with patch('requests.Session.request', return_value=SimpleNamespace(status_code=200)):
        second.get('https://data.alpaca.markets/v2/stocks/bars')
    assert now[0] >= limited_at + 60
    with patch('requests.Session.request', return_value=SimpleNamespace(
            status_code=429, headers={'Retry-After': '120'})):
        with pytest.raises(RuntimeError, match='stop downloader'):
            first.get('https://data.alpaca.markets/v2/stocks/bars')


def test_default_sessions_share_quota():
    assert BoundedSession().pacer is BoundedSession().pacer

"""CLI date boundaries must not accidentally ingest a later research interval."""
from pathlib import Path
from datetime import date
import sys

import pytest

from rl_stocks.data.storage.immutable import load
from rl_stocks.runtime.local import main
from rl_stocks.core.contracts import Request
from rl_stocks.data.accumulation.backfill import accumulate
from rl_stocks.data.providers.synthetic.adapter import SyntheticProvider
from rl_stocks.data.validation.bars import exchange_sessions


def test_ingest_uses_requested_exclusive_date_bounds(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['rl-stocks', 'ingest', '--provider', 'synthetic',
        '--start', '2020-01-01', '--end', '2020-03-01', '--full-range', '--output', str(tmp_path)])
    main()
    snapshot = Path(capsys.readouterr().out.strip())
    bars, provenance = load(snapshot)
    assert provenance['request']['start'] == '2020-01-01'
    assert provenance['request']['end'] == '2020-03-01'
    assert set(provenance['request']['symbols']) == {'SPY', 'QQQ', 'IWM', 'DIA'}
    assert all('2020-01-01' <= bar.session < '2020-03-01' for bar in bars)
    assert max(bar.session for bar in bars) == '2020-02-28'


def test_partial_collection_retains_failed_symbols_and_storage_errors(tmp_path):
    class PartialProvider(SyntheticProvider):
        def bars(self, request):
            if request.symbols == ('QQQ',):
                raise ValueError('private-provider-response-must-not-be-persisted')
            sessions = set(exchange_sessions(request))
            return [bar for bar in super().bars(request) if bar.session in sessions]

    request = Request(('SPY', 'QQQ'), date(2020, 1, 1), date(2020, 3, 1))
    snapshot = accumulate(PartialProvider(), request, tmp_path / 'partial',
                          full_range=True, continue_on_error=True)
    bars, provenance = load(snapshot)
    assert {bar.symbol for bar in bars} == {'SPY'}
    assert provenance['request']['symbols'] == ['SPY', 'QQQ']
    assert provenance['coverage']['QQQ']['count'] == 0
    assert provenance['coverage']['QQQ']['missing'] == exchange_sessions(request)
    assert provenance['acquisition_failures'] == [{'symbol': 'QQQ', 'start': '2020-01-01',
                                                  'end': '2020-03-01', 'error_type': 'ValueError'}]
    assert 'private-provider-response' not in str(provenance)
    with pytest.raises(ValueError, match='private-provider-response'):
        accumulate(PartialProvider(), request, tmp_path / 'strict', full_range=True)
    part = tmp_path / 'partial' / provenance['parts'][0] / 'snapshot.json'
    part.write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        accumulate(PartialProvider(), request, tmp_path / 'partial',
                   full_range=True, continue_on_error=True)

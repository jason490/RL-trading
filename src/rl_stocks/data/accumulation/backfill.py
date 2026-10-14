"""Resumable immutable year/symbol chunks. Refresh explicitly creates new revisions."""
from dataclasses import asdict
from datetime import date
import hashlib
import json
from pathlib import Path
import time
from rl_stocks.core.contracts import RecordType, Request, digest
from rl_stocks.data.storage.immutable import put, load
from rl_stocks.data.validation.bars import coverage, exchange_sessions


def accumulate(provider, request: Request, root, verify_calendar=True, refresh=False,
               history_starts=None, full_range=False):
    provider.capabilities.require(RecordType.BARS)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    identity = {"capabilities": asdict(provider.capabilities)}
    if hasattr(provider, 'seed'):
        identity['seed'] = provider.seed
    if hasattr(provider, 'path'):
        identity['file_hash'] = hashlib.sha256(provider.path.read_bytes()).hexdigest()
    key = digest(identity)
    cursor = root / ('cursor-' + key + '.json')
    progress = json.loads(cursor.read_text()) if cursor.exists() else {}
    bars, parts = [], []
    history_starts = history_starts or {}
    for symbol in request.symbols:
        intervals = [(request.start, request.end)] if full_range else [
            (max(request.start, date(year, 1, 1)), min(request.end, date(year+1, 1, 1)))
            for year in range(request.start.year, request.end.year+1)]
        for start, end in intervals:
            if symbol in history_starts:
                start = max(start, date.fromisoformat(history_starts[symbol]))
            if start >= end:
                continue
            part_request = Request((symbol,), start, end)
            chunk = f'{symbol}-{start}-{end}'
            if chunk in progress and not refresh:
                path = root / progress[chunk]
                part_bars, _ = load(path)
            else:
                # SDK-level retries remain bounded as well; no feed fallback.
                for attempt in range(3):
                    try:
                        part_bars = list(provider.bars(part_request))
                        break
                    except Exception as exc:
                        transient=isinstance(exc,(TimeoutError,ConnectionError)) or type(exc).__name__ in (
                            "Timeout","ConnectTimeout","ReadTimeout","ConnectionError","YFRateLimitError")
                        if not transient:
                            raise
                        if attempt == 2:
                            raise
                        time.sleep(attempt + 1)
                part_provenance = {**identity, 'request': {'symbol': symbol, 'start': str(start), 'end': str(end)}}
                path = put(root, part_bars, part_provenance, getattr(provider, 'raw_source', None))
                progress[chunk] = path.name
                temporary = cursor.with_suffix('.tmp')
                temporary.write_text(json.dumps(progress, indent=2))
                temporary.replace(cursor)
            bars.extend(part_bars)
            parts.append(path.name)
    bars.sort(key=lambda b: (b.symbol, b.session))
    expected = exchange_sessions(request) if verify_calendar else None
    report = coverage(bars, request, expected)
    for symbol, start in history_starts.items():
        if symbol in report:
            report[symbol]["missing"] = [day for day in report[symbol]["missing"] if day >= start]
            report[symbol]["declared_history_start"] = start
    provenance = {'request': {'symbols': request.symbols, 'start': str(request.start), 'end': str(request.end)},
        'capabilities': asdict(provider.capabilities), 'coverage': report,
        'parts': parts, 'dividends_verified': provider.capabilities.provider != 'alpaca'}
    return put(root, bars, provenance, {'immutable_parts': parts})

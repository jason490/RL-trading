"""Read-only daily Yahoo collection; preserve immutable receipts and raw responses."""
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from rl_stocks.core.contracts import Request, utcnow
from rl_stocks.data.providers.yahoo.adapter import YahooProvider
from rl_stocks.data.storage.immutable import load, put
from rl_stocks.data.validation.bars import timestamp
from rl_stocks.inference.shadow.journal import transaction, get
from rl_stocks.inference.shadow.protocol import read_protocol


def collect(root, output, *, now=None, provider=None):
    current = timestamp(now or utcnow())
    protocol = read_protocol(root)
    if protocol['feed_contract']['provider'] != 'yahoo':
        raise ValueError('automatic collection supports Yahoo only; import an audited snapshot')
    # Overlap audits the most recently observed prices. Full historical revisions
    # outside this window are not claimed to be detected by this collection step.
    with transaction(Path(root)/'journal.sqlite3') as db:
        paths = get(db, 'snapshots', [])
    source = paths[-1] if paths else protocol['snapshot']
    bars, _ = load(source)
    last = min(max(b.session for b in bars if b.symbol == s)
               for s in protocol['required_symbols'])
    request = Request(tuple(protocol['required_symbols']),
                      date.fromisoformat(last)-timedelta(days=7),
                      current.date()+timedelta(days=1))
    provider = provider or YahooProvider()
    received = provider.bars(request)
    # Restated split-adjusted prices cannot be mixed with existing positions.
    for response in getattr(provider, 'raw_source', []):
        for row in response['rows']:
            if row.get('Stock Splits', 0) and row['date'][:10] >= last:
                raise ValueError('new Yahoo split requires adjustment audit; shadow halted')
    complete = [b for b in received if timestamp(b.available_time) <= current
                and timestamp(b.retrieved_time) <= current]
    if not complete:
        raise ValueError('no completed available bars')
    return put(output, complete, {'kind': 'shadow-receipt', 'campaign': Path(root).name,
                                 'observed_at': current.isoformat(), 'request': {
                                     'symbols': request.symbols, 'start': str(request.start),
                                     'end': str(request.end)},
                                 'capabilities': asdict(provider.capabilities)},
               raw_source=getattr(provider, 'raw_source', None))

"""Acquire/cache the fixed old-ETF panel and audit it; never merge or train."""
import argparse,json
from datetime import date
from pathlib import Path
from rl_stocks.core.contracts import Request,digest
from rl_stocks.data.accumulation.backfill import accumulate
from rl_stocks.data.storage.immutable import load
from rl_stocks.data.providers.yahoo.adapter import YahooProvider
from rl_stocks.data.validation.bars import exchange_sessions
from rl_stocks.data.validation.historical import audit_history


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--acquire',action='store_true');p.add_argument('--snapshot',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new output path required')
    protocol=json.loads(Path('configs/experiments/historical_regime_audit.json').read_text())
    req=Request(tuple(protocol['symbols']),date.fromisoformat(protocol['start']),date.fromisoformat(protocol['end_exclusive']))
    if req.start!=date(2001,1,1) or req.end!=date(2010,1,1) or req.symbols!=('SPY','QQQ','IWM','DIA'):raise ValueError('fixed training-only audit contract required')
    if a.acquire==bool(a.snapshot):raise ValueError('choose acquire or an existing snapshot')
    snapshot=accumulate(YahooProvider(),req,'data/raw/historical-regimes',full_range=True) if a.acquire else a.snapshot
    bars,provenance=load(snapshot)
    if provenance['request']!={'symbols':list(req.symbols),'start':str(req.start),'end':str(req.end)}:raise ValueError('snapshot request mismatch')
    parts=[]
    for name in provenance['parts']:
        part=snapshot.parent/name;load(part)
        parts.extend(json.loads((part/'snapshot.json').read_text())['raw_source'])
    result=audit_history(bars,parts,req.symbols,exchange_sessions(req))
    result.update(snapshot=snapshot.name,protocol_sha256=digest(protocol),test_evaluated=False,colab_allocations=0)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Accepted four source-consistent historical ETF panels; no merge, fitting or Colab.')


if __name__=='__main__':main()

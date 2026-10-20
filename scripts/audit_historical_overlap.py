"""Fixed source-audited seam comparison; never rescale, merge or train."""
import argparse,json
from datetime import date
from pathlib import Path
from rl_stocks.core.contracts import Request,digest
from rl_stocks.data.accumulation.backfill import accumulate
from rl_stocks.data.providers.yahoo.adapter import YahooProvider
from rl_stocks.data.storage.immutable import load
from rl_stocks.data.validation.overlap import compare_overlap
from compare_historical_regimes import checked_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--acquire',action='store_true');p.add_argument('--snapshot',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new output path required')
    if a.acquire==bool(a.snapshot):raise ValueError('choose acquire or cached snapshot')
    protocol=json.loads(Path('configs/experiments/historical_overlap.json').read_text());checks=protocol['checks']
    req=Request(tuple(protocol['symbols']),date.fromisoformat(protocol['start']),date.fromisoformat(protocol['end_exclusive']))
    if req!=Request(('SPY','QQQ','IWM','DIA'),date(2009,12,1),date(2010,4,1)):raise ValueError('fixed seam-only request required')
    source=accumulate(YahooProvider(),req,'data/raw/historical-overlap',full_range=True) if a.acquire else a.snapshot
    source_audit=checked_snapshot(source,req);bars,_=load(source)
    references=[('earlier',Path('data/raw/historical-regimes')/protocol['earlier_snapshot'],date(2009,12,1),date(2010,1,1)),
        ('recent',Path('data/raw/universe-expansion')/protocol['recent_snapshot'],date(2010,1,1),date(2010,4,1))]
    results={}
    for name,path,start,end in references:
        request=Request(req.symbols,start,end);reference_audit=checked_snapshot(path,request)
        reference,_=load(path);select=lambda seq:[b for b in seq if b.symbol in req.symbols and str(start)<=b.session<str(end)]
        results[name]=compare_overlap(select(reference),select(bars),price_atol=checks['price_absolute_tolerance'],price_rtol=checks['price_relative_tolerance'],dividend_atol=checks['dividend_absolute_tolerance'])
        split_events=lambda audit:{r['symbol']:[e for e in r['source_splits'] if str(start)<=e['session']<str(end)] for r in audit['records']}
        same_splits=split_events(reference_audit)==split_events(source_audit)
        results[name]['raw_split_events_match']=same_splits
        results[name]['passed']=results[name]['passed'] and same_splits
    report=dict(protocol_sha256=digest(protocol),snapshot=source.name,passed=all(v['passed'] for v in results.values()),comparisons=results,
        source_audit=source_audit,training_performed=False,automatic_merge=False,colab_allocations=0,test_evaluated=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Overlap passed:',report['passed'],'; no merge, fitting or Colab.')
    if not report['passed']:raise SystemExit(2)


if __name__=='__main__':main()

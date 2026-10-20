"""Compare fixed historical training snapshots using identical past-only counts."""
import argparse,json
from pathlib import Path
from rl_stocks.core.contracts import Request
from rl_stocks.data.storage.immutable import load
from rl_stocks.data.validation.bars import exchange_sessions
from rl_stocks.data.validation.historical import audit_history
from datetime import date


def checked_snapshot(path,request):
    bars,provenance=load(path)
    bars=[b for b in bars if b.symbol in request.symbols and str(request.start)<=b.session<str(request.end)]
    raw=[]
    for name in provenance['parts']:
        part=path.parent/name
        # Verify immutable child before using its retained original rows.
        _,meta=load(part)
        if meta['request']['symbol'] not in request.symbols:continue
        for source in json.loads((part/'snapshot.json').read_text())['raw_source']:
            raw.append(dict(symbol=source['symbol'],rows=[r for r in source['rows'] if str(request.start)<=str(r['date'])[:10]<str(request.end)]))
    return audit_history(bars,raw,request.symbols,exchange_sessions(request))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new output path required')
    old_audit=json.loads(Path('artifacts/historical-regimes/audit.json').read_text());symbols=('SPY','QQQ','IWM','DIA')
    old=Path('data/raw/historical-regimes')/old_audit['snapshot']
    recent=Path('data/raw/universe-expansion/be5a8d56ab667294ac741ba0ad55f9d25e391cd2740811f965ad77543d9c283f')
    result=dict(training_performed=False,automatic_merge=False,test_evaluated=False,regimes={})
    for name,path,start,end in [('earlier',old,2001,2010),('recent',recent,2010,2021)]:
        report=checked_snapshot(path,Request(symbols,date(start,1,1),date(end,1,1)))
        counts={k:sum(v[k] for r in report['records'] for v in r['regime_years'].values()) for k in ['rows','ready63','negative63','ready252','negative252','joint_negative']}
        result['regimes'][name]=dict(snapshot=path.name,request_start=f'{start}-01-01',end_exclusive=f'{end}-01-01',totals=counts,
            joint_decline_fraction=counts['joint_negative']/counts['ready252'],records=report['records'])
    result['interpretation']='Same past-only rule; unequal calendar lengths reported as fractions. No future losing-stock labels, new performance claim or causal benefit established.'
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v['joint_decline_fraction'] for k,v in result['regimes'].items()})


if __name__=='__main__':main()

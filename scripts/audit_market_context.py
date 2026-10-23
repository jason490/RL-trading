"""Fixed cached-data availability audit; no fit, inference or outcome scoring."""
import argparse,json
from dataclasses import asdict
from pathlib import Path
from collections import Counter
from rl_stocks.core.contracts import digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.data.storage.immutable import load
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.training.diversity import validate_payload
from rl_stocks.features.market.context import MarketContext,NAMES

PROTOCOL_SHA256='4d610a3c133b94ebe74426a9ab21d66c55304f1d7945fc7d69eb3d4c06ae44b0'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new evidence output required')
    protocol=Path('configs/experiments/market_context_audit.json');spec=json.loads(protocol.read_text())
    if digest(spec)!=PROTOCOL_SHA256 or list(NAMES)!=spec['names']:raise ValueError('fixed market-context audit changed')
    source=Path('data/datasets')/spec['source_dataset'];raw=Path('data/raw/universe-expansion')/spec['raw_snapshot']
    files=[protocol,source/'dataset.json',raw/'snapshot.json',Path('scripts/audit_market_context.py'),Path('src/rl_stocks/features/market/context.py')];protected={str(p):checksum(p) for p in files}
    data=validate_payload(read_dataset(source),'expanded');bars,provenance=load(raw)
    if data['manifest']['snapshot']!=raw.name or any(b.session>='2024-01-01' for b in bars):raise ValueError('exact pre2024 cached source required')
    symbols={e['symbol'] for e in data['episodes'].values()};coverage=provenance['coverage']
    if any(coverage[s]['calendar_verified'] is not True or coverage[s]['missing'] for s in symbols):
        raise ValueError('complete audited source calendars required')
    bars=[b for b in bars if b.symbol in symbols];context=MarketContext(bars,{s:coverage[s]['missing'] for s in symbols})
    lookup={(b.symbol,b.session):b for b in bars};counts=Counter();unavailable=[];per_symbol={}
    for key,e in data['episodes'].items():
        ready=0
        for b in e['bars']:
            if asdict(lookup[(b['symbol'],b['session'])])!=b:raise ValueError('raw/source bar mismatch')
            value=context.at(e['symbol'],b['session'],b['available_time'])
            counts[e['split']+'_rows']+=1
            if value is None:unavailable.append(dict(episode=key,session=b['session']))
            else:ready+=1;counts[e['split']+'_ready']+=1
        per_symbol[key]=dict(symbol=e['symbol'],split=e['split'],rows=len(e['bars']),ready=ready)
    if any(checksum(Path(p))!=h for p,h in protected.items()):raise ValueError('source changed')
    result=dict(passed=not unavailable,protocol_sha256=PROTOCOL_SHA256,source_dataset=source.name,raw_snapshot=raw.name,
        names=list(NAMES),benchmark='SPY',window=20,counts=dict(counts),episodes=per_symbol,unavailable=unavailable,
        raw_source_bars_exact=True,source_files_unchanged=True,protected_sha256=protected,training_performed=False,
        model_inference=False,normalization_fitted=False,provider_requested=False,colab_allocated=False,
        test_evaluated=False,validation_performance_scored=False,interpretation=spec['limitations'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Coverage audit:',dict(counts),'unavailable:',len(unavailable),'No fitting/performance scoring.')


if __name__=='__main__':main()

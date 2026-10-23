"""Build immutable opt-in market payloads; no policy fitting or Colab allocation."""
import argparse,json,math
from pathlib import Path
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.core.files import write_once
from rl_stocks.inference.bundle import checksum
from rl_stocks.data.storage.immutable import load
from rl_stocks.data.generation.datasets import read_dataset,normalizer_from
from rl_stocks.data.generation.market_context import build_payload,validate_payload_pair
from rl_stocks.features.market.context import MarketContext
from rl_stocks.training.diversity import validate_payload

PREREGISTRATION_SHA256='ca987adbfa0355b843cd9e0a545e9695db4b1bc046f5c63d6e83dc85ad0d52c7'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new preflight output required')
    protocol=Path('configs/experiments/market_context_preregistration.json');spec=json.loads(protocol.read_text())
    if digest(spec)!=PREREGISTRATION_SHA256:raise ValueError('fixed market-context preregistration changed')
    audit=Path('results/market-context-2026-10-23/coverage.json')
    if checksum(audit)!=spec['audit_evidence_sha256'] or json.loads(audit.read_text()).get('passed') is not True:
        raise ValueError('exact passing coverage evidence required')
    root=Path('data/datasets');source=read_dataset(root/spec['source_dataset']);validate_payload(source,'expanded')
    raw=Path('data/raw/universe-expansion')/spec['raw_snapshot'];bars,provenance=load(raw)
    if source['manifest']['snapshot']!=raw.name or any(b.session>='2024-01-01' for b in bars):raise ValueError('same pre2024 snapshot required')
    symbols={e['symbol'] for e in source['episodes'].values()};coverage=provenance['coverage']
    if any(coverage[s]['calendar_verified'] is not True or coverage[s]['missing'] for s in symbols):
        raise ValueError('complete verified source calendars required')
    context=MarketContext([b for b in bars if b.symbol in symbols],{s:coverage[s]['missing'] for s in symbols})
    files=[protocol,audit,raw/'snapshot.json',root/spec['source_dataset']/'dataset.json',Path('scripts/prepare_market_context.py'),Path('src/rl_stocks/data/generation/market_context.py')]
    protected={str(p):checksum(p) for p in files};records={}
    for mode in ('masked','full'):
        payload=build_payload(source,context,mode);identity=digest(payload);target=root/identity;target.mkdir(exist_ok=True)
        write_once(target/'dataset.json',canonical(payload));restored=read_dataset(target)
        validate_payload_pair(source,restored,context,mode)
        normalizer=normalizer_from(restored['normalizer']);maximum=0.
        for e in restored['episodes'].values():
            for row in e['features']:
                transformed=normalizer.transform(row,restored['names'])
                if not all(math.isfinite(v) and abs(v)<=3.4028234663852886e38 for v in transformed):
                    raise ValueError('finite float32 model inputs required')
                if mode=='masked' and transformed[7:]!=[0.]*3:raise ValueError('exact normalized mask required')
                maximum=max(maximum,max(abs(v) for v in transformed))
        records[mode]=dict(dataset=identity,schema=restored['schema'],normalizer_sha256=digest(restored['normalizer']),
            normalizer=restored['normalizer'],raw_bars_atr_original_features_and_rows_exact=True,
            extra_normalizer_fit_rows=restored['manifest']['market_context']['extra_normalizer_fit_rows'],
            maximum_abs_normalized_value=maximum,finite_float32_inputs=True)
    if any(checksum(Path(p))!=h for p,h in protected.items()):raise ValueError('source/preregistration changed')
    result=dict(passed=True,preregistration_sha256=PREREGISTRATION_SHA256,source_dataset=spec['source_dataset'],
        records=records,source_files_unchanged=True,protected_sha256=protected,
        training_rows=sum(len(e['bars']) for e in source['episodes'].values() if e['split']=='train'),
        validation_rows=sum(len(e['bars']) for e in source['episodes'].values() if e['split']=='validation'),
        policy_training_performed=False,extra_normalizer_fitted_on_training_only=True,
        validation_used_for_fit=False,provider_requested=False,colab_allocated=False,test_evaluated=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Both immutable payloads verified; original source unchanged. No policy fitting/allocation.')


if __name__=='__main__':main()

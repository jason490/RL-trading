"""Immutable cap5 payload and legacy identity check; no policy training."""
import argparse,json
from pathlib import Path
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import generate,read_dataset,normalizer_from
from rl_stocks.data.generation.clipping import clipped_dataset,validate_clipped_dataset
from rl_stocks.training.diversity import validate_payload

SOURCE='cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b'
DEFAULT='c2e6c58c4d4b0a9853b5b37851c731f559f42e597e9eefcf77796d874db31398'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new evidence output required')
    root=Path('data/datasets');source=validate_payload(read_dataset(root/SOURCE),'expanded');m=source['manifest']
    default=generate(Path('data/raw/universe-expansion')/m['snapshot'],root,
        universe=m['universe_config'],evaluation_symbols=m['evaluation_symbols'],include_test=False,
        coverage_scope='universe',purge=m['purge_sessions'])
    if default.name!=DEFAULT:raise ValueError('legacy default generation identity changed')
    candidate=clipped_dataset(source);identity=digest(candidate);destination=root/identity;destination.mkdir(exist_ok=True)
    write_once(destination/'dataset.json',canonical(candidate));candidate=validate_clipped_dataset(source,read_dataset(destination))
    normal=normalizer_from(candidate['normalizer']);largest=0.
    for e in candidate['episodes'].values():
        for row in e['features']:largest=max(largest,max(abs(v) for v in normal.transform(row,candidate['names'])))
    if largest>5 or digest(read_dataset(root/SOURCE))!=SOURCE:raise ValueError('clipping bound/source changed')
    result=dict(passed=True,source_dataset=SOURCE,candidate_dataset=identity,clip_abs=5.,schema=candidate['schema'],
        normalizer_sha256=digest(candidate['normalizer']),source_normalizer_sha256=digest(source['normalizer']),
        raw_episodes_sha256=digest(source['episodes']),legacy_default_generated_identity=default.name,
        legacy_default_identity_unchanged=True,raw_episodes_features_bars_atr_exact=candidate['episodes']==source['episodes'],
        base_means_scales_unchanged=all(candidate['normalizer'][k]==source['normalizer'][k] for k in ['names','means','scales']),
        maximum_transformed_market_abs=largest,training_symbols=len({e['symbol'] for e in source['episodes'].values() if e['split']=='train'}),
        training_rows=sum(len(e['bars']) for e in source['episodes'].values() if e['split']=='train'),validation_rows=sum(len(e['bars']) for e in source['episodes'].values() if e['split']=='validation'),
        candidate_normalization_fitted=False,legacy_training_normalizer_recomputed_for_identity_check=True,training_performed=False,provider_requested=False,colab_allocated=False,test_evaluated=False,source_unchanged=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

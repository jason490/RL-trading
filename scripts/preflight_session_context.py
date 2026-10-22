"""Prepare fixed development-only matched session decomposition datasets; no training."""
import argparse,json
from pathlib import Path
from statistics import mean
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import generate,read_dataset
from rl_stocks.data.generation.session_context import preserve_preprocessing
from rl_stocks.data.storage.immutable import load
from rl_stocks.features.price.sessions import NAMES

SOURCE='cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b'
GENERATED='c2e6c58c4d4b0a9853b5b37851c731f559f42e597e9eefcf77796d874db31398'
SNAPSHOT='09bc69ec15798c6bf2987c3332397104ce3066846bad638b1512e29e41f1028f'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new preflight output required')
    root=Path('data/datasets');source=read_dataset(root/SOURCE);m=source['manifest']
    snapshot=Path('data/raw/universe-expansion')/SNAPSHOT
    if m['snapshot']!=SNAPSHOT or m['provenance']['request']['end']!='2024-01-01':raise ValueError('fixed development source required')
    bars,provenance=load(snapshot)
    if provenance!=m['provenance'] or any(b.session>'2023-12-31' for b in bars):raise ValueError('development-only snapshot mismatch')
    del bars
    options=dict(universe=m['universe_config'],evaluation_symbols=m['evaluation_symbols'],include_test=False,
        coverage_scope='universe',purge=m['purge_sessions'])
    default=generate(snapshot,root,**options)
    if default.name!=GENERATED:raise ValueError('default generated identity changed')
    records=dict(legacy=dict(dataset=SOURCE,schema=source['schema'],names=source['names'],normalizer_sha256=digest(source['normalizer'])))
    panel=digest({k:dict(bars=e['bars'],atr=e['atr']) for k,e in source['episodes'].items()})
    for mode in ['masked','full']:
        generated=read_dataset(generate(snapshot,root,session_mode=mode,**options))
        data=preserve_preprocessing(source,generated,mode);identity=digest(data);destination=root/identity;destination.mkdir(exist_ok=True)
        write_once(destination/'dataset.json',canonical(data));read_dataset(destination)
        records[mode]=dict(dataset=identity,generated_dataset=digest(generated),schema=data['schema'],names=data['names'],
            normalizer_sha256=digest(data['normalizer']),training_rows=sum(len(e['bars']) for e in data['episodes'].values() if e['split']=='train'),
            validation_rows=sum(len(e['bars']) for e in data['episodes'].values() if e['split']=='validation'))
        if mode=='full':
            rows=[r[-4:] for e in data['episodes'].values() if e['split']=='train' for r in e['features']]
            diagnostic={n:dict(minimum=min(c),mean=mean(c),maximum=max(c)) for n,c in zip(NAMES,zip(*rows))}
    result=dict(passed=True,source_dataset=SOURCE,source_snapshot=SNAPSHOT,datasets=records,default_generated_dataset_identity_unchanged=True,
        all_source_bars_atr_and_original_features_exact=True,matched_bars_atr_sha256=panel,original_seven_normalization_frozen=True,
        training_only_feature_diagnostic=diagnostic,training_performed=False,test_evaluated=False,sessions_allocated=0,
        limitations='Retrospective survivors, revised Yahoo data and repeatedly reused development; input coverage is not predictive evidence')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(records,indent=2))


if __name__=='__main__':main()

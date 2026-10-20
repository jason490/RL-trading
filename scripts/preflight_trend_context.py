"""Build matched causal context datasets from a fixed development-only snapshot."""
import argparse
import json
from pathlib import Path
from statistics import mean
from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import read_dataset, generate
from rl_stocks.features.trend.context import NAMES

SOURCE='672a21621091a56c1c7beb30557464e7edd8df38874494659bab2c83738e1bf9'
SNAPSHOT='be5a8d56ab667294ac741ba0ad55f9d25e391cd2740811f965ad77543d9c283f'


def prepare(data_root,snapshot,output):
    original=read_dataset(data_root/SOURCE);manifest=original['manifest']
    if snapshot.name!=SNAPSHOT or manifest['snapshot']!=SNAPSHOT:
        raise ValueError('fixed development snapshot required')
    if (manifest['provenance']['request']['end']!='2024-01-01'
        or manifest.get('included_splits')!=['train','validation']
        or any(e['split']=='test' or any(b['session']>'2023-12-31' for b in e['bars']) for e in original['episodes'].values())):
        raise ValueError('verified development-only input required')
    from rl_stocks.data.storage.immutable import load
    source_bars,source_provenance=load(snapshot)
    if (source_provenance!=manifest['provenance'] or any(b.session>'2023-12-31' for b in source_bars)):
        raise ValueError('source provenance or development cutoff mismatch')
    source_end=max(b.session for b in source_bars)
    del source_bars
    options=dict(universe=manifest['universe_config'],evaluation_symbols=manifest['evaluation_symbols'],
        include_test=False,purge=manifest['purge_sessions'])
    default=generate(snapshot,data_root,**options)
    if default.name!=SOURCE:raise ValueError('default dataset identity changed')
    datasets={};panel=None;training_panel=None;base_stats=None;records={}
    for mode in ('matched_baseline','masked','full'):
        path=generate(snapshot,data_root,trend_context=mode,**options);d=read_dataset(path)
        raw={k:{'bars':e['bars'],'atr':e['atr']} for k,e in d['episodes'].items()}
        val=digest({k:v for k,v in raw.items() if d['episodes'][k]['split']=='validation'})
        train=digest({k:v for k,v in raw.items() if d['episodes'][k]['split']=='train'})
        stats={k:d['normalizer'][k][:7] for k in ('means','scales')}
        if panel is not None and (val!=panel or train!=training_panel or stats!=base_stats):
            raise ValueError('matched panels or baseline normalization differ')
        panel=val;training_panel=train;base_stats=stats
        if {e['split'] for e in d['episodes'].values()}!={'train','validation'}:raise ValueError('split isolation failed')
        for key,e in d['episodes'].items():
            if e['split']=='validation' and e['bars']!=original['episodes'][key]['bars']:
                raise ValueError('original validation execution dates changed')
        datasets[mode]=path.name
        records[mode]={'dataset':path.name,'schema':d['schema'],'names':d['names'],
            'normalizer_sha256':digest(d['normalizer']),
            'training_rows':sum(len(e['bars']) for e in d['episodes'].values() if e['split']=='train'),
            'validation_rows':sum(len(e['bars']) for e in d['episodes'].values() if e['split']=='validation')}
        if mode=='full':
            rows=[r[-3:] for e in d['episodes'].values() if e['split']=='train' for r in e['features']]
            diagnostic={name:{'minimum':min(c),'mean':mean(c),'maximum':max(c)} for name,c in zip(NAMES,zip(*rows))}
            diagnostic['annual_and_quarterly_negative_fraction']=mean(r[0]<0 and r[1]<0 for r in rows)
    result={'passed':True,'source_dataset':SOURCE,'source_snapshot':SNAPSHOT,'default_dataset_identity_unchanged':True,'source_sessions_end':source_end,
        'datasets':records,'matched_train_bars_atr_sha256':training_panel,'matched_validation_bars_atr_sha256':panel,
        'baseline_normalization_identical_across_arms':True,'original_validation_bars_unchanged':True,
        'training_only_context_diagnostic':diagnostic,'normalization_fit':'Each dataset training rows only; first7 statistics identical',
        'training_performed':False,'test_evaluated':False,'sessions_allocated':0,
        'limits':'Descriptive input coverage, not evidence of future predictive power or profitability'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'datasets':datasets,'training_performed':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('data/datasets'))
    p.add_argument('--snapshot',type=Path,default=Path('data/raw/universe-expansion')/SNAPSHOT)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.data_root,a.snapshot,a.output)

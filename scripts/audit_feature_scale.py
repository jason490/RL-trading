"""Audit frozen market-feature scales on fixed cached train/development sources."""
import argparse,json,math
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.feature_scale import summarize_scale
from rl_stocks.training.diversity import validate_payload

SOURCE='cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b'


def audit():
    root=Path.cwd();path=root/'data/datasets'/SOURCE;original=checksum(path/'dataset.json')
    data=validate_payload(read_dataset(path),'expanded');normalizer=digest(data['normalizer'])
    if data['manifest']['included_splits']!=['train','validation']:raise ValueError('train/development-only source required')
    panel=set(data['manifest']['evaluation_symbols']);stocks={i['symbol'] for i in data['manifest']['universe_config']['instruments'] if i['kind']=='stock'}
    episodes=list(data['episodes'].values());definitions={
        'training_all':lambda e:e['split']=='train',
        'training_original_panel':lambda e:e['split']=='train' and e['symbol'] in panel,
        'training_additional_symbols':lambda e:e['split']=='train' and e['symbol'] not in panel,
        'development_original_panel':lambda e:e['split']=='validation' and e['symbol'] in panel}
    records={}
    for label,choose in definitions.items():
        records[label]={}
        for kind in ['stock','etf']:
            selected=[e for e in episodes if choose(e) and (e['symbol'] in stocks)==(kind=='stock')]
            if not selected:continue
            rows=[r for e in selected for r in e['features']]
            records[label][kind]=dict(symbol_count=len({e['symbol'] for e in selected}),row_count=len(rows),
                features=summarize_scale(rows,data['names'],data['normalizer']))
    extremes={}
    for name in ['return_1','atr_14_ratio','relative_volume_20']:
        j=data['names'].index(name);mu=data['normalizer']['means'][j];scale=data['normalizer']['scales'][j]
        ranked=[]
        for e in episodes:
            if e['split']!='train':continue
            for i,row in enumerate(e['features']):
                z=(row[j]-mu)/scale
                if abs(z)>20:ranked.append((abs(z),e['symbol'],i,e,row[j],z))
        ranked.sort(key=lambda r:(-r[0],r[1],r[2]));examples=[]
        for _,symbol,i,e,value,z in ranked[:3]:
            bar=e['bars'][i];recomputed=None
            if name=='return_1' and i>0:recomputed=math.log(bar['close']/e['bars'][i-1]['close'])
            if name=='atr_14_ratio':recomputed=e['atr'][i]/bar['close']
            if name=='relative_volume_20' and i>=20:recomputed=bar['volume']/(sum(b['volume'] for b in e['bars'][i-20:i])/20)
            if recomputed is not None and not math.isclose(value,recomputed,abs_tol=1e-12,rel_tol=1e-12):raise ValueError('extreme feature/source arithmetic mismatch')
            examples.append(dict(symbol=symbol,session=bar['session'],feature_value=value,normalized_value=z,
                recomputed_from_source=recomputed,arithmetic_matches=recomputed is not None,
                note='Source arithmetic only; not independent provider or corporate-action verification'))
        extremes[name]=examples
    if checksum(path/'dataset.json')!=original or digest(data['normalizer'])!=normalizer:raise ValueError('immutable source or normalization changed')
    return dict(dataset=SOURCE,dataset_file_sha256=original,normalizer_sha256=normalizer,records=records,training_extreme_examples=extremes,
        source_sha256={p:checksum(root/p) for p in ['scripts/audit_feature_scale.py','src/rl_stocks/reporting/feature_scale.py']},
        source_unchanged=True,normalization_unchanged=True,normalization_fitted=False,training_performed=False,
        model_opened=False,provider_requested=False,colab_allocated=False,test_evaluated=False,objective_achieved=False,
        interpretation='Available-row distributions, not sampled transition mass or predictive evidence. Float64 frozen z values before float32 observation cast. Fixed abs3/5/10 thresholds descriptive, no clipping or outcome-based selection.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new output required')
    result=audit();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Audited fixed market feature distributions without outcomes/model fitting/Colab.')

"""Prove training-only target folds and preprocessing before Colab model fits."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from rl_stocks.core.contracts import digest,canonical
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.training.supervised.targets import completed_targets,fit_preprocessing,HORIZON,FIT_END,PROBE_START,PROBE_END


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();protocol=json.loads(Path('configs/experiments/supervised_signal.json').read_text())
    if (protocol['arms']!=['masked','full'] or protocol['target']['horizon_sessions']!=HORIZON
        or protocol['folds']['fit_label_completed_through']!=FIT_END
        or protocol['folds']['probe_decision_start']!=PROBE_START
        or protocol['folds']['probe_label_completed_through']!=PROBE_END):raise ValueError('fixed target/fold protocol required')
    data=read_dataset(Path('data/datasets')/protocol['source_dataset'])
    if len(data['names'])!=10 or protocol['source_features']!=10:raise ValueError('fixed10-field source required')
    rows=completed_targets(data)
    stats={};normalizers={}
    for arm in protocol['arms']:
        transformed=rows if arm=='full' else ({**r,'features':r['features'][:7]+[0.,0.,0.]} for r in rows)
        normalizers[arm]=asdict(fit_preprocessing(transformed,data['names']))
    assert all(normalizers['masked'][key][:7]==normalizers['full'][key][:7] for key in ['means','scales'])
    for fold in ['fit','probe']:
        selected=[r for r in rows if r['fold']==fold]
        stats[fold]={'count':len(selected),'symbols':len({r['symbol'] for r in selected}),
            'decision_start':min(r['decision_session'] for r in selected),'decision_end':max(r['decision_session'] for r in selected),
            'last_label_available':max(r['label_available_time'] for r in selected),
            'rows_sha256':digest(selected),'year_counts':{year:sum(r['decision_session'].startswith(year) for r in selected) for year in sorted({r['decision_session'][:4] for r in selected})}}
    payload={'schema':'completed-training-signal-v1','source_dataset':protocol['source_dataset'],
        'names':data['names'],'rows':rows,'fold_normalizers':normalizers,
        'target_contract':{'horizon_sessions':20,'fit_label_end':'2017-12-31',
            'probe_decision_start':'2018-01-01','probe_label_end':'2020-12-31',
            'entry':'next open after corporate actions','exit':'20th close including subsequent splits/dividends',
            'kind':'gross log unit total return, no reinvestment/costs/protection'}}
    signal_id=digest(payload)
    if protocol.get('signal_dataset',signal_id)!=signal_id:raise ValueError('prepared signal identity drift')
    destination=Path('data/signals')/signal_id/'dataset.json'
    destination.parent.mkdir(parents=True,exist_ok=True);write_once(destination,canonical(payload))
    assert read_dataset(destination.parent)==json.loads(canonical(payload))
    result={'signal_dataset':signal_id,'passed':True,'protocol_sha256':digest(protocol),'source_dataset':protocol['source_dataset'],
        'folds':stats,'normalizers':normalizers,'same_first7_fit_statistics':True,
        'source_pooled_normalizer_used':False,'project_validation_used':False,'test_evaluated':False,
        'model_training_performed':False,'colab_allocated':False,'target_kind':'Gross completed20-session unit total return, not tradable P&L'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['passed','folds','model_training_performed','colab_allocated']}))


if __name__=='__main__':main()

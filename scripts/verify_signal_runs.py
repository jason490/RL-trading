"""Restore returned forecasts against immutable probe inputs without fitting."""
import argparse,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from rl_stocks.core.contracts import canonical
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.inference.signal import SignalBundle
from rl_stocks.training.supervised.spec import SIGNAL_DATASET,validate_payload


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    data=validate_payload(read_dataset(Path('data/signals')/SIGNAL_DATASET));probe=[r for r in data['rows'] if r['fold']=='probe']
    raw=np.asarray([r['features'] for r in probe]);target=np.asarray([r['target_log_total_return'] for r in probe])
    paths=list((a.root/'runs').glob('*/metadata.json'))
    if len(paths)!=1:raise ValueError('one supervised run required')
    path=paths[0];m=json.loads(path.read_text());bundle=SignalBundle(a.root/'frozen'/m['bundle_id'],data['names'],data['source_dataset'])
    if bundle.manifest['normalizers']!=data['fold_normalizers']:raise ValueError('fit preprocessing mismatch')
    before=canonical({arm:asdict(n) for arm,n in bundle.normalizers.items()});records=[]
    with np.load(path.parent/'probe-predictions.npz',allow_pickle=False) as arrays:
        np.testing.assert_array_equal(target,arrays['target'])
        np.testing.assert_array_equal([r['decision_session'] for r in probe],arrays['sessions'])
        np.testing.assert_array_equal([r['symbol'] for r in probe],arrays['symbols'])
        for arm in ('masked','full'):
            for seed in (11,23,37):
                restored=bundle.predict(raw,arm,seed);repeat=bundle.predict(raw,arm,seed)
                np.testing.assert_array_equal(restored,repeat)
                np.testing.assert_allclose(restored,arrays[f'{arm}-{seed}'],rtol=0.,atol=1e-6)
                records.append({'arm':arm,'seed':seed,'predictions':len(restored),'deterministic_repeat_exact':True,
                    'max_remote_prediction_error':float(np.max(np.abs(restored-arrays[f'{arm}-{seed}'])))})
        for arm in ('masked','full'):
            for kind,key in [('constant','constant'),('ridge','ridge_'+arm)]:np.testing.assert_allclose(bundle.predict(raw,arm,kind=kind),arrays[key],rtol=0,atol=1e-10)
    assert canonical({arm:asdict(n) for arm,n in bundle.normalizers.items()})==before
    for names,source in [(list(reversed(data['names'])),data['source_dataset']),(data['names'],'wrong-source')]:
        try:SignalBundle(a.root/'frozen'/m['bundle_id'],names,source)
        except ValueError:pass
        else:raise ValueError('signal schema accepted')
    try:bundle.predict([[float('nan')]*10],'full')
    except ValueError:pass
    else:raise ValueError('nonfinite input accepted')
    result={'passed':True,'bundle_id':m['bundle_id'],'normalization_matches_immutable_fit_data':True,
        'probe_target_session_symbol_arrays_exact':True,'normalization_unchanged':True,
        'schema_source_nonfinite_rejection':True,'records':records,'training_performed':False,'project_validation_used':False,'test_evaluated':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print('Verified',len(records),'forecasts on',len(probe),'immutable probe rows; no fitting.')


if __name__=='__main__':main()

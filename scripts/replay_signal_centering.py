"""Preregistered fixed-forecast diagnostic; no training or trading simulation."""
import argparse
import copy
from io import BytesIO
import json
from pathlib import Path
import numpy as np
from rl_stocks.core.contracts import digest,canonical
from rl_stocks.core.provenance import repository_state
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.inference.bundle import checksum
from rl_stocks.inference.signal import SignalBundle
from rl_stocks.inference.signal_centering import center_forecasts
from rl_stocks.training.supervised.artifacts import verify_report
from rl_stocks.training.supervised.metrics import score,gate

RECIPE='centered_i,t = raw_mlp_i,t - mean_26(raw_mlp_j,t) + mean_26(same_arm_frozen_ridge_j,t)'


def prepare(protocol):
    if (protocol['recipe']!=RECIPE or protocol['arms']!=['masked','full'] or protocol['seeds']!=[11,23,37]
        or protocol['constraints']['eligible_for_promotion'] is not False
        or protocol['constraints']['original_gate_remains_failed'] is not True):raise ValueError('fixed diagnostic protocol required')
    state=json.loads(Path('.runtime/signal-feasibility-state.json').read_text())
    if state['status']!='complete' or not state['cleanup_verified']:raise ValueError('completed source and cleanup required')
    root=Path(state['completed']['signal-feasibility']['root']);paths=list((root/'runs').glob('*/metadata.json'))
    if len(paths)!=1:raise ValueError('one source run required')
    run=paths[0].parent;metadata=json.loads(paths[0].read_text());result=json.loads((run/'result.json').read_text())
    published=Path(protocol['source_failure']);proof_path=Path('results/supervised-signal-2026-10-20/restoration.json')
    proof=json.loads(proof_path.read_text())
    if result['gate']['passed'] or checksum(published)!=checksum(run/'result.json') or not proof['passed'] or proof['bundle_id']!=protocol['source_bundle']:raise ValueError('verified original failure required')
    data_path=Path('data/signals')/protocol['source_signal_dataset']/'dataset.json';data=read_dataset(data_path.parent)
    probe=[r for r in data['rows'] if r['fold']=='probe'];bundle_path=root/'frozen'/protocol['source_bundle']
    bundle=SignalBundle(bundle_path,data['names'],data['source_dataset'])
    if metadata['bundle_id']!=protocol['source_bundle'] or bundle.manifest['normalizers']!=data['fold_normalizers']:raise ValueError('source bundle/normalizer mismatch')
    verify_report(lambda name:(run/name).read_bytes(),'',metadata,bundle.manifest)
    prediction_path=run/'probe-predictions.npz'
    moments=json.loads(Path('results/supervised-signal-2026-10-20/forecast-moments.json').read_text())
    if checksum(prediction_path)!=moments['source_prediction_sha256']:raise ValueError('cached prediction identity drift')
    with np.load(prediction_path,allow_pickle=False) as source:arrays={k:source[k].copy() for k in source.files}
    np.testing.assert_array_equal(arrays['target'],[r['target_log_total_return'] for r in probe])
    np.testing.assert_array_equal(arrays['sessions'],[r['decision_session'] for r in probe]);np.testing.assert_array_equal(arrays['symbols'],[r['symbol'] for r in probe])
    universe_path=Path(protocol['universe']);universe=json.loads(universe_path.read_text());symbols=[i['symbol'] for i in universe['instruments'] if i['role']=='tradable']
    if len(symbols)!=26:raise ValueError('fixed26-symbol universe required')
    available=[r['feature_available_time'] for r in probe];entry=[r['entry_session'] for r in probe]
    transformed={};panels=None
    for arm in protocol['arms']:
        for seed in protocol['seeds']:
            key=f'{arm}-{seed}';transformed[key],p=center_forecasts(arrays[key],arrays['ridge_'+arm],arrays['sessions'],arrays['symbols'],available,entry,symbols)
            if panels is not None and p!=panels:raise ValueError('availability panels differ')
            panels=p
    files={'metadata':paths[0],'source_result':run/'result.json','source_predictions':prediction_path,
        'published_failure':published,'restoration':proof_path,'prepared_signal':data_path,'universe':universe_path,
        **{'bundle/'+p.name:p for p in bundle_path.iterdir() if p.is_file()}}
    protected={name:checksum(p) for name,p in files.items()}
    proof={'protocol_sha256':digest(protocol),'source_files_sha256':protected,'availability_panels_sha256':digest(panels),
        'sessions':len(panels),'symbols_per_session':26,'rows':len(probe),'first_cutoff':panels[0],
        'last_cutoff':panels[-1],'original_gate_passed':False,'training_performed':False,'colab_allocated':False,'project_validation_used':False,'test_evaluated':False}
    return proof,arrays,transformed,result,files


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight',type=Path,default=Path('results/signal-centering-2026-10-20/preflight.json'))
    a=p.parse_args();protocol=json.loads(Path('configs/experiments/signal_centering.json').read_text())
    proof,arrays,transformed,original,files=prepare(protocol)
    if a.preflight_only:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(canonical(proof));print('Availability/identity preflight passed; no transformed performance scored.');return
    source=repository_state(Path.cwd())
    if source['dirty']:raise ValueError('committed clean source required before scoring')
    if proof!=json.loads(a.preflight.read_text()):raise ValueError('committed preflight drift')
    if a.output.exists():raise ValueError('new diagnostic output required')
    metrics=copy.deepcopy(original['metrics']);pairs={}
    for arm in protocol['arms']:
        for seed in protocol['seeds']:
            key=f'{arm}-{seed}';old=original['metrics']['mlp'][arm][str(seed)]
            new=score(arrays['target'],transformed[key],arrays['sessions']);metrics['mlp'][arm][str(seed)]=new
            if abs(old['mean_daily_spearman_ic']-new['mean_daily_spearman_ic'])>1e-12:raise ValueError('within-session rank drift')
            pairs[key]={'raw_mse':old['mse'],'centered_mse':new['mse'],'mse_reduction_from_raw':1-new['mse']/old['mse'],'daily_rank_correlation_preserved':True}
    if any(checksum(p)!=proof['source_files_sha256'][k] for k,p in files.items()):raise ValueError('source mutation')
    result={'protocol':protocol,'source':source,'preflight':proof,'metrics':metrics,'paired_comparisons':pairs,
        'descriptive_source_numeric_gate':gate(metrics),'original_gate_passed':False,'eligible_for_promotion':False,
        'independent_confirmation':False,'profitability_established':False,'source_files_unchanged':True,
        'training_performed':False,'colab_allocated':False,'project_validation_used':False,'test_evaluated':False}
    a.output.mkdir(parents=True);(a.output/'summary.json').write_bytes(canonical(result))
    np.savez_compressed(a.output/'centered-predictions.npz',**transformed)
    print(json.dumps(result['descriptive_source_numeric_gate'],indent=2))


if __name__=='__main__':main()

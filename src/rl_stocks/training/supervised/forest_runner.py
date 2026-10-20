"""Fixed Colab-only forests on completed training-era targets."""
from dataclasses import asdict
from datetime import datetime,timezone
import json,os,shutil,uuid
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.core.provenance import repository_state,source_identity
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.inference.bundle import checksum
from rl_stocks.inference.forest_arrays import export_forest,load_forest,predict_forest
from rl_stocks.inference.forest_signal import ForestSignalBundle
from rl_stocks.training.supervised.spec import validate_payload
from rl_stocks.training.supervised.forest_spec import protocol,make_forest,verify_forest_environment
from rl_stocks.training.supervised.models import design,ridge_fit,ridge_predict
from rl_stocks.training.supervised.metrics import score,gate
from rl_stocks.training.supervised.runner import snapshot


def run(dataset_path,output):
    if os.environ.get('RL_COLAB_OWNED_WORKER')!='1':raise RuntimeError('forest fitting requires the owned Colab worker')
    config=protocol();data=validate_payload(read_dataset(dataset_path));versions=verify_forest_environment()
    output=Path(output);run=output/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-forest-'+uuid.uuid4().hex[:8]);run.mkdir(parents=True);stage=run/'models';stage.mkdir()
    fit=[r for r in data['rows'] if r['fold']=='fit'];probe=[r for r in data['rows'] if r['fold']=='probe']
    raw=np.asarray([r['features'] for r in fit]);raw_probe=np.asarray([r['features'] for r in probe]);y=np.asarray([r['target_log_total_return'] for r in fit]);target=np.asarray([r['target_log_total_return'] for r in probe]);days=np.asarray([r['decision_session'] for r in probe])
    normalizers=data['fold_normalizers'];before=canonical(normalizers);x={a:design(raw,normalizers[a],a) for a in ('masked','full')};xp={a:design(raw_probe,normalizers[a],a) for a in x}
    for bad in (np.zeros((1,9)),np.full((1,10),float('nan'))):
        try:design(bad,normalizers['full'],'full')
        except ValueError:pass
        else:raise ValueError('prefix input rejection failed')
    with threadpool_limits(limits=1):
        prefix=make_forest(11,True);prefix.fit(x['full'][:512],y[:512]);export_forest(prefix,run/'prefix-check.npz')
        np.testing.assert_allclose(prefix.predict(xp['full'][:512]),predict_forest(load_forest(run/'prefix-check.npz',2),xp['full'][:512]),rtol=0,atol=1e-10)
        constant=float(y.mean());ridge={a:ridge_fit(x[a],y).tolist() for a in x}
        predictions={'constant':np.full(len(target),constant)};metrics={'constant':score(target,predictions['constant'],days),'ridge':{},'forest':{a:{} for a in x}}
        weights={};seeds={};curves=[];parity={}
        for arm in ('masked','full'):
            predictions['ridge_'+arm]=ridge_predict(ridge[arm],xp[arm]);metrics['ridge'][arm]=score(target,predictions['ridge_'+arm],days)
            for seed in (11,23,37):
                model=make_forest(seed);model.fit(x[arm],y);key=f'{arm}-{seed}';weights[key]=key+'.npz';export_forest(model,stage/weights[key])
                predictions[key]=model.predict(xp[arm]);portable=predict_forest(load_forest(stage/weights[key],128),xp[arm])
                np.testing.assert_allclose(predictions[key],portable,rtol=0,atol=1e-10)
                parity[key]=float(np.max(np.abs(predictions[key]-portable)));seeds[key]=digest([e.random_state for e in model.estimators_])
                metrics['forest'][arm][str(seed)]=score(target,predictions[key],days)
                fit_mse=float(np.mean((model.predict(x[arm])-y)**2));curves.append({'arm':arm,'seed':seed,'trees':128,'fit_mse':fit_mse})
                (run/'fit-progress.json').write_bytes(canonical({'completed':list(weights),'fit_summaries':curves}));snapshot(output)
    assert all(seeds[f'masked-{s}']==seeds[f'full-{s}'] for s in (11,23,37))
    metadata={'algorithm':'forest_signal_feasibility','seed':11,'model_seeds':[11,23,37],'dataset':Path(dataset_path).name,
        'requested_steps':770,'actual_steps':770,'work_unit':'fitted trees including2 discarded prefix trees','smoke':False,'device':'cpu','status':'complete',
        'requirements_lock_sha256':checksum(Path(__file__).resolve().parents[4]/'requirements.lock'),
        'source_hash':source_identity(),'git':repository_state(Path(__file__).resolve().parents[4]),'versions':versions,'protocol_sha256':digest(config),
        'fit_rows':len(fit),'probe_rows':len(probe),'paired_tree_seeds_match':True,'tree_seed_fingerprints':seeds,
        'prefix_input_rejection':True,'prefix_portable_parity':True,'normalization_unchanged':canonical(normalizers)==before,'portable_max_errors':parity}
    manifest={'kind':ForestSignalBundle.KIND,'names':data['names'],'source_dataset':data['source_dataset'],'signal_dataset':Path(dataset_path).name,
        'normalizers':normalizers,'weights':weights,'ridge':ridge,'constant':constant,'protocol':config,'metadata':metadata}
    identity=digest(manifest);bundle=output/'frozen'/identity;bundle.mkdir(parents=True)
    for name in weights.values():shutil.copyfile(stage/name,bundle/name)
    for filename in ('requirements.lock','requirements.txt'):shutil.copyfile(Path(__file__).resolve().parents[4]/filename,bundle/filename)
    (bundle/'dependencies.json').write_bytes(canonical(versions))
    (bundle/'manifest.json').write_bytes(canonical(manifest));(bundle/'checksums.json').write_bytes(canonical({p.name:checksum(p) for p in bundle.iterdir()}))
    frozen=ForestSignalBundle(bundle,data['names'],data['source_dataset']);frozen_before=canonical({a:asdict(n) for a,n in frozen.normalizers.items()})
    for arm in x:
        for seed in (11,23,37):
            pred=frozen.predict(raw_probe,arm,seed);np.testing.assert_allclose(pred,predictions[f'{arm}-{seed}'],rtol=0,atol=1e-10);np.testing.assert_array_equal(pred,frozen.predict(raw_probe,arm,seed))
    for names,source in [(list(reversed(data['names'])),data['source_dataset']),(data['names'],'wrong-source')]:
        try:ForestSignalBundle(bundle,names,source)
        except ValueError:pass
        else:raise ValueError('forest schema/source rejection failed')
    try:frozen.predict([[float('nan')]*10],'full')
    except ValueError:pass
    else:raise ValueError('forest nonfinite input accepted')
    assert canonical({a:asdict(n) for a,n in frozen.normalizers.items()})==frozen_before
    result={'metrics':metrics,'gate':gate(metrics,'forest'),'fixed_tree_count':128,'prefix_portable_parity':True,'frozen_prediction_parity':True,
        'normalization_unchanged':True,'schema_source_nonfinite_rejection':True,'training_performed_in_colab':True,
        'project_validation_used':False,'test_evaluated':False,'profitability_established':False}
    (run/'result.json').write_bytes(canonical(result));(run/'fit-summary.json').write_bytes(canonical(curves))
    np.savez_compressed(run/'probe-predictions.npz',target=target,sessions=days,symbols=np.asarray([r['symbol'] for r in probe]),**predictions)
    metadata['bundle_id']=identity;(run/'metadata.json').write_bytes(canonical(metadata));snapshot(output)
    return run,bundle

"""Bounded Colab CPU fitting; no trading, project-validation or test scoring."""
from dataclasses import asdict
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import tarfile
import uuid
import numpy as np
import torch
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.core.provenance import repository_state,source_identity
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.inference.bundle import checksum
from rl_stocks.inference.signal import SignalBundle
from rl_stocks.training.dependencies import verify_environment
from rl_stocks.training.supervised.spec import protocol,validate_payload
from rl_stocks.training.supervised.models import network,design,predict,ridge_fit,ridge_predict
from rl_stocks.training.supervised.metrics import score,gate


def snapshot(output):
    target=os.environ.get('RL_CHECKPOINT_ARCHIVE')
    if target:
        target=Path(target);temporary=target.with_suffix('.tmp')
        with tarfile.open(temporary,'w:gz') as t:t.add(output,arcname='artifacts')
        temporary.replace(target)


def run(dataset_path,output):
    if os.environ.get('RL_COLAB_OWNED_WORKER')!='1':raise RuntimeError('supervised fitting requires the owned Colab worker')
    config=protocol();data=validate_payload(read_dataset(dataset_path));versions=verify_environment();torch.set_num_threads(1)
    output=Path(output);name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-signal-'+uuid.uuid4().hex[:8]
    run=output/'runs'/name;run.mkdir(parents=True);stage=run/'models';stage.mkdir()
    fit=[r for r in data['rows'] if r['fold']=='fit'];probe=[r for r in data['rows'] if r['fold']=='probe']
    raw_fit=np.asarray([r['features'] for r in fit],dtype=np.float64);raw_probe=np.asarray([r['features'] for r in probe],dtype=np.float64)
    y=np.asarray([r['target_log_total_return'] for r in fit]);target=np.asarray([r['target_log_total_return'] for r in probe]);sessions=np.asarray([r['decision_session'] for r in probe])
    names=data['names'];normalizers=data['fold_normalizers'];before=canonical(normalizers)
    x={arm:design(raw_fit,normalizers[arm],arm) for arm in ('masked','full')};xp={arm:design(raw_probe,normalizers[arm],arm) for arm in ('masked','full')}
    for bad in (np.zeros((1,9)),np.full((1,10),float('nan'))):
        try:design(bad,normalizers['full'],'full')
        except ValueError:pass
        else:raise ValueError('prefix input contract accepted invalid data')
    # Tiny prefix acceptance is discarded; full fits reseed from scratch.
    torch.manual_seed(11);tiny=network();opt=torch.optim.Adam(tiny.parameters(),lr=.001)
    for _ in range(2):
        opt.zero_grad();loss=((tiny(torch.tensor(x['full'][:32],dtype=torch.float32)).squeeze(-1)-torch.tensor(y[:32],dtype=torch.float32))**2).mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite prefix loss')
        loss.backward();opt.step()
    torch.save(tiny.state_dict(),run/'prefix-check.pt');restored=network();restored.load_state_dict(torch.load(run/'prefix-check.pt',weights_only=True,map_location='cpu'))
    np.testing.assert_array_equal(predict(tiny,xp['full'][:3]),predict(restored,xp['full'][:3]))
    constant=float(y.mean());ridge={a:ridge_fit(x[a],y).tolist() for a in x}
    predictions={'constant':np.full(len(target),constant)};metrics={'constant':score(target,predictions['constant'],sessions),'ridge':{},'mlp':{a:{} for a in x}}
    weights={};initial={};curves=[];spec=config['models']['mlp']
    for arm in ('masked','full'):
        predictions['ridge_'+arm]=ridge_predict(ridge[arm],xp[arm]);metrics['ridge'][arm]=score(target,predictions['ridge_'+arm],sessions)
        for seed in (11,23,37):
            torch.manual_seed(seed);model=network();initial[f'{arm}-{seed}']=digest({k:v.detach().numpy().tolist() for k,v in model.state_dict().items()})
            optimizer=torch.optim.Adam(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay']);rng=np.random.default_rng(seed)
            xt=torch.as_tensor(x[arm],dtype=torch.float32);yt=torch.as_tensor(y,dtype=torch.float32)
            for update in range(1,spec['updates']+1):
                index=rng.integers(len(y),size=spec['batch_size']);optimizer.zero_grad();loss=((model(xt[index]).squeeze(-1)-yt[index])**2).mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite supervised loss')
                loss.backward();optimizer.step()
                if update%200==0:curves.append({'arm':arm,'seed':seed,'update':update,'fit_minibatch_mse':float(loss.detach())})
            if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite signal weights')
            key=f'{arm}-{seed}';weights[key]=key+'.pt';torch.save(model.state_dict(),stage/weights[key])
            predictions[key]=predict(model,xp[arm]);metrics['mlp'][arm][str(seed)]=score(target,predictions[key],sessions)
            (run/'fit-progress.json').write_bytes(canonical({'completed':list(weights),'curves':curves}));snapshot(output)
    assert all(initial[f'masked-{s}']==initial[f'full-{s}'] for s in (11,23,37))
    metadata={'algorithm':'signal_feasibility','seed':11,'model_seeds':[11,23,37],'dataset':Path(dataset_path).name,
        'requested_steps':12002,'actual_steps':12002,'work_unit':'optimizer updates including2 discarded prefix steps',
        'smoke':False,'device':'cpu','status':'complete','source_hash':source_identity(),
        'git':repository_state(Path(__file__).resolve().parents[4]),'versions':versions,'protocol_sha256':digest(config),
        'fit_rows':len(fit),'probe_rows':len(probe),'initial_tensors_match':True,'initial_fingerprints':initial,
        'prefix_input_rejection':True,'prefix_save_load_exact':True,'normalization_unchanged':canonical(normalizers)==before}
    manifest={'kind':'signal-feasibility-v1','names':names,'source_dataset':data['source_dataset'],
        'signal_dataset':Path(dataset_path).name,'normalizers':normalizers,'weights':weights,'ridge':ridge,
        'constant':constant,'protocol':config,'metadata':metadata}
    identity=digest(manifest);bundle=output/'frozen'/identity;bundle.mkdir(parents=True)
    for filename in weights.values():shutil.copyfile(stage/filename,bundle/filename)
    (bundle/'manifest.json').write_bytes(canonical(manifest));(bundle/'checksums.json').write_bytes(canonical({p.name:checksum(p) for p in bundle.iterdir()}))
    frozen=SignalBundle(bundle,names,data['source_dataset']);normalizer_before=canonical({a:asdict(n) for a,n in frozen.normalizers.items()})
    for arm in ('masked','full'):
        for seed in (11,23,37):np.testing.assert_array_equal(predictions[f'{arm}-{seed}'],frozen.predict(raw_probe,arm,seed))
    for bad_names,bad_source in [(list(reversed(names)),data['source_dataset']),(names,'incompatible-source')]:
        try:SignalBundle(bundle,bad_names,bad_source)
        except ValueError:pass
        else:raise ValueError('signal schema rejection failed')
    try:frozen.predict([[float('nan')]*10],'full')
    except ValueError:pass
    else:raise ValueError('nonfinite forecast input accepted')
    assert canonical({a:asdict(n) for a,n in frozen.normalizers.items()})==normalizer_before
    result={'metrics':metrics,'gate':gate(metrics),'fixed_final_update':spec['updates'],
        'prefix_save_load_exact':True,'frozen_prediction_parity':True,'normalization_unchanged':True,
        'schema_source_nonfinite_rejection':True,'training_performed_in_colab':True,
        'project_validation_used':False,'test_evaluated':False,'profitability_established':False}
    (run/'result.json').write_bytes(canonical(result));(run/'training-loss.json').write_bytes(canonical(curves))
    np.savez_compressed(run/'probe-predictions.npz',target=target,sessions=sessions,symbols=np.asarray([r['symbol'] for r in probe]),**predictions)
    metadata['bundle_id']=identity;(run/'metadata.json').write_bytes(canonical(metadata));snapshot(output)
    return run,bundle

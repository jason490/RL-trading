from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.inference.signal import SignalBundle
from rl_stocks.runtime.colab.jobs import signal_plan,validate_plan
from rl_stocks.training.supervised.models import network,design,ridge_predict
from rl_stocks.training.supervised.metrics import score,gate


def test_predictive_metrics_gate_years_seeds_and_tie_order():
    y=np.tile(np.arange(10)*.01,3);days=np.repeat(['2018-01-02','2019-01-02','2020-01-02'],10)
    constant=score(y,np.full(len(y),y.mean()),days)
    assert constant['undefined_ic_days']==3 and constant['mean_daily_spearman_ic']==0
    good=score(y,y+.001,days);ridge=score(y,y+.01,days)
    assert good['mean_daily_spearman_ic']==pytest.approx(1.)
    m={'constant':constant,'ridge':{a:ridge for a in ['masked','full']},
       'mlp':{a:{str(s):good for s in [11,23,37]} for a in ['masked','full']}}
    result=gate(m);assert result['passed'] and result['candidate']=='masked'
    assert not result['profitability_established']
    changed=deepcopy(m)
    for a in ['masked','full']:
        for seed in ['11','23']:changed['mlp'][a][seed]=constant
    assert not gate(changed)['passed']
    with pytest.raises(ValueError):score(y,y,['2024-01-02']*len(y))
    with pytest.raises(ValueError):score(y,np.full(len(y),float('nan')),days)


def test_predictor_restores_without_training_and_rejects_input_or_file_changes(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(11);model=network();normalizer={'names':[str(i) for i in range(10)],'means':[0.]*10,'scales':[1.]*10}
    stage=tmp_path/'stage';stage.mkdir();weights={}
    for arm in ['masked','full']:
        for seed in [11,23,37]:
            name=f'{arm}-{seed}.pt';weights[f'{arm}-{seed}']=name;torch.save(model.state_dict(),stage/name)
    manifest={'kind':'signal-feasibility-v1','names':normalizer['names'],'source_dataset':'fixture',
        'normalizers':{a:normalizer for a in ['masked','full']},'weights':weights,
        'constant':.01,'ridge':{a:[.01]+[0.]*10 for a in ['masked','full']}}
    bundle=tmp_path/digest(manifest);stage.rename(bundle)
    (bundle/'manifest.json').write_bytes(canonical(manifest));(bundle/'checksums.json').write_bytes(canonical({p.name:checksum(p) for p in bundle.iterdir()}))
    restored=SignalBundle(bundle,normalizer['names'],'fixture');raw=np.ones((3,10))
    before=deepcopy(restored.manifest)
    np.testing.assert_array_equal(restored.predict(raw,'full'),restored.predict(raw,'full'))
    np.testing.assert_array_equal(restored.predict(raw,'full',kind='constant'),[.01]*3)
    assert restored.manifest==before
    for names,source in [(normalizer['names'],'wrong'),(list(reversed(normalizer['names'])),'fixture')]:
        with pytest.raises(ValueError):SignalBundle(bundle,names,source)
    with pytest.raises(ValueError):restored.predict(np.full((1,10),float('nan')),'full')
    with pytest.raises(ValueError):restored.predict(raw,'full',seed=99)
    with pytest.raises(ValueError):restored.predict(np.ones((1,11)),'full')
    (bundle/'full-11.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError):SignalBundle(bundle,normalizer['names'],'fixture')


def test_masks_preserve_inputs_and_seeded_initial_models_and_plan_is_fixed():
    raw=np.arange(20).reshape(2,10);before=raw.copy();norm={'means':[0.]*10,'scales':[1.]*10}
    x=design(raw,norm,'masked');np.testing.assert_array_equal(x[:,7:],0.);np.testing.assert_array_equal(raw,before)
    torch.manual_seed(23);a=network();torch.manual_seed(23);b=network()
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
    plan=signal_plan();assert validate_plan(plan)==plan
    changed=deepcopy(plan);changed['jobs'][0]['steps']=50000
    with pytest.raises(ValueError):validate_plan(changed)
    from rl_stocks.training.supervised.runner import run
    with pytest.raises(RuntimeError,match='owned Colab'):run('missing','unused')


def test_artifact_metrics_recomputed_from_predictions_reject_tampering():
    from io import BytesIO
    from rl_stocks.training.supervised.artifacts import verify_report
    from rl_stocks.training.supervised.spec import protocol,PROTOCOL_SHA256,SIGNAL_DATASET
    from datetime import date,timedelta
    days=np.array([str(date(year,1,1)+timedelta(days=i//26)) for year,n in [(2018,6526),(2019,6552),(2020,5538)] for i in range(n)])
    y=np.linspace(-.1,.1,len(days));arrays={'target':y,'sessions':days,'symbols':np.array([f'S{i%26}' for i in range(len(y))]),
        'constant':np.full(len(y),y.mean()),'ridge_masked':y+.01,'ridge_full':y+.01,
        **{f'{a}-{s}':y+.001 for a in ['masked','full'] for s in [11,23,37]}}
    m={'constant':score(y,arrays['constant'],days),'ridge':{},'mlp':{}}
    for arm in ['masked','full']:
        m['ridge'][arm]=score(y,arrays['ridge_'+arm],days);m['mlp'][arm]={str(s):score(y,arrays[f'{arm}-{s}'],days) for s in [11,23,37]}
    result={'metrics':m,'gate':gate(m),'fixed_final_update':2000,**{k:True for k in ['prefix_save_load_exact','frozen_prediction_parity','normalization_unchanged','schema_source_nonfinite_rejection','training_performed_in_colab']},
        **{k:False for k in ['project_validation_used','test_evaluated','profitability_established']}}
    meta={'protocol_sha256':PROTOCOL_SHA256,'actual_steps':12002,'requested_steps':12002,'dataset':SIGNAL_DATASET,
        'initial_fingerprints':{f'{a}-{s}':'0'*64 for a in ['masked','full'] for s in [11,23,37]},'fit_rows':45266,'probe_rows':18616,'initial_tensors_match':True,'prefix_input_rejection':True,'prefix_save_load_exact':True,'normalization_unchanged':True}
    manifest={'protocol':protocol(),'signal_dataset':SIGNAL_DATASET}
    buffer=BytesIO();np.savez_compressed(buffer,**arrays)
    files={'result.json':canonical(result),'probe-predictions.npz':buffer.getvalue(),
        'training-loss.json':canonical([{'arm':a,'seed':s,'update':u,'fit_minibatch_mse':.01} for a in ['masked','full'] for s in [11,23,37] for u in range(200,2001,200)])}
    assert verify_report(files.__getitem__,'',meta,manifest)['gate']['passed']
    changed=deepcopy(result);changed['metrics']['mlp']['full']['11']['mse']=0.
    files['result.json']=canonical(changed)
    with pytest.raises(RuntimeError,match='recomputation'):verify_report(files.__getitem__,'',meta,manifest)
    files['result.json']=canonical(result);changed=deepcopy(result);changed['project_validation_used']=True;files['result.json']=canonical(changed)
    with pytest.raises(RuntimeError,match='scope'):verify_report(files.__getitem__,'',meta,manifest)

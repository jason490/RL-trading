from copy import deepcopy
from datetime import date,timedelta
from io import BytesIO
from pathlib import Path
import numpy as np
import pytest
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.inference.forest_arrays import export_forest
from rl_stocks.inference.forest_signal import ForestSignalBundle
from rl_stocks.runtime.colab.jobs import forest_plan,validate_plan
from rl_stocks.training.supervised.metrics import score,gate
from test_forest_arrays import tree
from types import SimpleNamespace


def test_forest_bundle_restores_repeatable_forecasts_and_checks_contracts(tmp_path):
    stub=SimpleNamespace(n_features_in_=10,n_outputs_=1,estimators_=[SimpleNamespace(tree_=SimpleNamespace(**tree())) for _ in range(128)])
    stage=tmp_path/'stage';stage.mkdir();weights={}
    for a in ('masked','full'):
        for s in (11,23,37):
            name=f'{a}-{s}.npz';export_forest(stub,stage/name);weights[f'{a}-{s}']=name
    norm={'names':[str(i) for i in range(10)],'means':[0.]*10,'scales':[1.]*10}
    manifest={'kind':ForestSignalBundle.KIND,'names':norm['names'],'source_dataset':'fixture',
        'normalizers':{a:norm for a in ('masked','full')},'weights':weights,'constant':.01,'ridge':{a:[.01]+[0.]*10 for a in ('masked','full')}}
    folder=tmp_path/digest(manifest);stage.rename(folder);(folder/'manifest.json').write_bytes(canonical(manifest));(folder/'checksums.json').write_bytes(canonical({p.name:checksum(p) for p in folder.iterdir()}))
    model=ForestSignalBundle(folder,norm['names'],'fixture');raw=np.zeros((3,10));raw[:,0]=[-1,0,1]
    np.testing.assert_allclose(model.predict(raw,'full'),[-.02,-.02,.03],atol=1e-15)
    np.testing.assert_array_equal(model.predict(raw,'full'),model.predict(raw,'full'))
    np.testing.assert_array_equal(model.predict(raw,'masked',kind='ridge'),[.01]*3)
    with pytest.raises(ValueError):model.predict(raw,'full',99)
    with pytest.raises(ValueError):model.predict([[float('nan')]*10],'full')
    with pytest.raises(ValueError):ForestSignalBundle(folder,list(reversed(norm['names'])),'fixture')
    (folder/'full-11.npz').write_bytes(b'bad')
    with pytest.raises(ValueError):ForestSignalBundle(folder,norm['names'],'fixture')


def test_forest_plan_locks_work_units_and_local_fitting_is_guarded():
    p=forest_plan();assert validate_plan(p)==p
    for mutate in [lambda p:p['jobs'][0].update(steps=768),lambda p:p.update(accelerator='A100')]:
        bad=deepcopy(p);mutate(bad)
        with pytest.raises(ValueError):validate_plan(bad)
    from rl_stocks.training.supervised.forest_runner import run
    with pytest.raises(RuntimeError,match='owned Colab'):run('missing','unused')


def test_forest_artifact_recomputes_metrics_and_rejects_claimed_success():
    from rl_stocks.training.supervised.forest_artifacts import verify_report
    from rl_stocks.training.supervised.forest_spec import protocol,PROTOCOL_SHA256,TREE_VERSIONS
    from rl_stocks.training.supervised.spec import SIGNAL_DATASET
    days=np.array([str(date(year,1,1)+timedelta(days=i//26)) for year,n in [(2018,6526),(2019,6552),(2020,5538)] for i in range(n)])
    y=np.linspace(-.1,.1,len(days));arrays={'target':y,'sessions':days,'symbols':np.array([f'S{i%26}' for i in range(len(y))]),'constant':np.full(len(y),y.mean()),'ridge_masked':y+.01,'ridge_full':y+.01,**{f'{a}-{s}':y+.001 for a in ('masked','full') for s in (11,23,37)}}
    metrics={'constant':score(y,arrays['constant'],days),'ridge':{},'forest':{}}
    for a in ('masked','full'):
        metrics['ridge'][a]=score(y,arrays['ridge_'+a],days);metrics['forest'][a]={str(s):score(y,arrays[f'{a}-{s}'],days) for s in (11,23,37)}
    result={'metrics':metrics,'gate':gate(metrics,'forest'),'fixed_tree_count':128,**{k:True for k in ['prefix_portable_parity','frozen_prediction_parity','normalization_unchanged','schema_source_nonfinite_rejection','training_performed_in_colab']},**{k:False for k in ['project_validation_used','test_evaluated','profitability_established']}}
    expected={f'{a}-{s}' for a in ('masked','full') for s in (11,23,37)}
    metadata={'protocol_sha256':PROTOCOL_SHA256,'actual_steps':770,'requested_steps':770,'dataset':SIGNAL_DATASET,'fit_rows':45266,'probe_rows':18616,'versions':{'trees':TREE_VERSIONS},'tree_seed_fingerprints':{k:'0'*64 for k in expected},'portable_max_errors':{k:0. for k in expected},**{k:True for k in ['paired_tree_seeds_match','prefix_input_rejection','prefix_portable_parity','normalization_unchanged']}}
    manifest={'protocol':protocol(),'signal_dataset':SIGNAL_DATASET};buf=BytesIO();np.savez_compressed(buf,**arrays)
    files={'result.json':canonical(result),'probe-predictions.npz':buf.getvalue(),'fit-summary.json':canonical([{'arm':a,'seed':s,'trees':128,'fit_mse':.01} for a in ('masked','full') for s in (11,23,37)])}
    assert verify_report(files.__getitem__,'',metadata,manifest)['gate']['passed']
    changed=deepcopy(result);changed['metrics']['forest']['full']['11']['mse']=0.;files['result.json']=canonical(changed)
    with pytest.raises(RuntimeError,match='recomputation'):verify_report(files.__getitem__,'',metadata,manifest)

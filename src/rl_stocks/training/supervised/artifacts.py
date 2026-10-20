"""Independently recompute forecast metrics/gates from bounded returned artifacts."""
from io import BytesIO
import json
import math
import numpy as np
from rl_stocks.core.contracts import digest
from rl_stocks.training.supervised.metrics import score,gate
from rl_stocks.training.supervised.spec import protocol,PROTOCOL_SHA256,SIGNAL_DATASET


def same(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,bool) or a is None or isinstance(a,str):return type(a)==type(b) and a==b
    return isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-8)


def verify_report(read,prefix,metadata,manifest):
    if (metadata['protocol_sha256']!=PROTOCOL_SHA256 or metadata['actual_steps']!=12002
        or metadata['requested_steps']!=12002 or metadata['dataset']!=SIGNAL_DATASET
        or metadata['fit_rows']!=45266 or metadata['probe_rows']!=18616
        or not metadata['initial_tensors_match'] or not metadata['prefix_save_load_exact'] or not metadata['prefix_input_rejection']
        or not metadata['normalization_unchanged']):raise RuntimeError('signal training contract mismatch')
    if manifest['protocol']!=protocol() or manifest['signal_dataset']!=SIGNAL_DATASET:raise RuntimeError('frozen signal protocol mismatch')
    expected_keys={f'{a}-{s}' for a in ('masked','full') for s in (11,23,37)}
    fingerprints=metadata.get('initial_fingerprints',{})
    if set(fingerprints)!=expected_keys or any(fingerprints[f'masked-{s}']!=fingerprints[f'full-{s}'] for s in (11,23,37)):
        raise RuntimeError('initial signal tensors mismatch')
    result=json.loads(read(prefix+'result.json'))
    if any(result.get(k) is not True for k in ['prefix_save_load_exact','frozen_prediction_parity','normalization_unchanged','schema_source_nonfinite_rejection','training_performed_in_colab']):raise RuntimeError('signal verification incomplete')
    if any(result.get(k) is not False for k in ['project_validation_used','test_evaluated','profitability_established']):raise RuntimeError('signal scope violation')
    with np.load(BytesIO(read(prefix+'probe-predictions.npz')),allow_pickle=False) as arrays:
        expected={'target','sessions','symbols','constant','ridge_masked','ridge_full',*[f'{a}-{s}' for a in ('masked','full') for s in (11,23,37)]}
        if set(arrays.files)!=expected or len(arrays['target'])!=18616:raise RuntimeError('signal prediction panel mismatch')
        if (len(set(arrays['symbols']))!=26 or len(set(zip(arrays['sessions'],arrays['symbols'])))!=18616):raise RuntimeError('signal symbol/session panel mismatch')
        metrics={'constant':score(arrays['target'],arrays['constant'],arrays['sessions']),'ridge':{},'mlp':{}}
        for a in ('masked','full'):
            metrics['ridge'][a]=score(arrays['target'],arrays['ridge_'+a],arrays['sessions'])
            metrics['mlp'][a]={str(s):score(arrays['target'],arrays[f'{a}-{s}'],arrays['sessions']) for s in (11,23,37)}
    if not same(result['metrics'],metrics) or not same(result['gate'],gate(metrics)) or result['fixed_final_update']!=2000:raise RuntimeError('signal metric/gate recomputation mismatch')
    loss=json.loads(read(prefix+'training-loss.json'))
    if len(loss)!=60 or {(r['arm'],r['seed'],r['update']) for r in loss}!={(a,s,u) for a in ('masked','full') for s in (11,23,37) for u in range(200,2001,200)} or any(not math.isfinite(r['fit_minibatch_mse']) for r in loss):raise RuntimeError('signal fit curve mismatch')
    return result

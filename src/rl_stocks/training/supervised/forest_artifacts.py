"""Independent forest metric/gate and fit-work verification."""
from io import BytesIO
import json,math
import numpy as np
from rl_stocks.training.supervised.artifacts import same
from rl_stocks.training.supervised.metrics import score,gate
from rl_stocks.training.supervised.spec import SIGNAL_DATASET
from rl_stocks.training.supervised.forest_spec import protocol,PROTOCOL_SHA256,TREE_VERSIONS


def verify_report(read,prefix,metadata,manifest):
    if (metadata['protocol_sha256']!=PROTOCOL_SHA256 or metadata['actual_steps']!=770 or metadata['requested_steps']!=770
        or metadata['dataset']!=SIGNAL_DATASET or metadata['fit_rows']!=45266 or metadata['probe_rows']!=18616
        or not all(metadata[k] is True for k in ['paired_tree_seeds_match','prefix_input_rejection','prefix_portable_parity','normalization_unchanged'])
        or metadata['versions']['trees']!=TREE_VERSIONS):raise RuntimeError('forest training contract mismatch')
    if manifest['protocol']!=protocol() or manifest['signal_dataset']!=SIGNAL_DATASET:raise RuntimeError('frozen forest protocol mismatch')
    expected={f'{a}-{s}' for a in ('masked','full') for s in (11,23,37)};fingerprints=metadata['tree_seed_fingerprints']
    if set(fingerprints)!=expected or any(fingerprints[f'masked-{s}']!=fingerprints[f'full-{s}'] for s in (11,23,37)):raise RuntimeError('paired forest seeds mismatch')
    if set(metadata['portable_max_errors'])!=expected or any(not 0<=v<=1e-10 for v in metadata['portable_max_errors'].values()):raise RuntimeError('forest portable parity mismatch')
    result=json.loads(read(prefix+'result.json'))
    if any(result.get(k) is not True for k in ['prefix_portable_parity','frozen_prediction_parity','normalization_unchanged','schema_source_nonfinite_rejection','training_performed_in_colab']):raise RuntimeError('forest verification incomplete')
    if any(result.get(k) is not False for k in ['project_validation_used','test_evaluated','profitability_established']):raise RuntimeError('forest scope violation')
    with np.load(BytesIO(read(prefix+'probe-predictions.npz')),allow_pickle=False) as arrays:
        if (set(arrays.files)!={'target','sessions','symbols','constant','ridge_masked','ridge_full',*expected}
            or len(arrays['target'])!=18616 or len(set(arrays['symbols']))!=26 or len(set(zip(arrays['sessions'],arrays['symbols'])))!=18616):raise RuntimeError('forest prediction panel mismatch')
        metrics={'constant':score(arrays['target'],arrays['constant'],arrays['sessions']),'ridge':{},'forest':{}}
        for arm in ('masked','full'):
            metrics['ridge'][arm]=score(arrays['target'],arrays['ridge_'+arm],arrays['sessions'])
            metrics['forest'][arm]={str(seed):score(arrays['target'],arrays[f'{arm}-{seed}'],arrays['sessions']) for seed in (11,23,37)}
    if not same(result['metrics'],metrics) or not same(result['gate'],gate(metrics,'forest')) or result['fixed_tree_count']!=128:raise RuntimeError('forest metric/gate recomputation mismatch')
    fit=json.loads(read(prefix+'fit-summary.json'))
    if (len(fit)!=6 or {(r['arm'],r['seed']) for r in fit}!={(a,s) for a in ('masked','full') for s in (11,23,37)}
        or any(r['trees']!=128 or not math.isfinite(r['fit_mse']) for r in fit)):raise RuntimeError('forest fit summary mismatch')
    return result

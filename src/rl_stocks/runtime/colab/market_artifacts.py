"""Market-context-specific retrieved artifact checks; common integrity checks stay shared."""
import json,math
from rl_stocks.core.contracts import digest


def validate_market_artifacts(read,prefix,folder,run,plan):
    from rl_stocks.training.market_context import PROTOCOL_SHA256,validate_record,modes,validate_execution_counts,validate_execution_report,protocol as recurrent_protocol,initialization_spec,initialization_entry
    from rl_stocks.training.diversity import validate_counts,sampling_summary
    from rl_stocks.features.registry.assemble import NAMES
    from rl_stocks.data.generation.market_context import validate_frozen_contract
    frozen=json.loads(read(folder+'manifest.json'));validate_frozen_contract(frozen);arm=plan['market_study']['arm'];job=plan['jobs'][0]
    duration=run.get('duration_seconds')
    if isinstance(duration,bool) or not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<=0:
        raise RuntimeError('positive actual market run duration required')
    if run.get('market_arm')!=arm:raise RuntimeError('actual market arm mismatch')
    validate_record(run['market_initialization'],arm,job['smoke'],initial=True,seed=job['seed'])
    validate_record(run['market_final'],arm,job['smoke'],seed=job['seed'])
    validate_record(run['market_selected'],arm,job['smoke'],selected_steps=run['selected_step'],seed=job['seed'])
    from rl_stocks.training.shared_initialization import read_initialization
    proof=initialization_spec(arm,job['seed'],job['smoke'])
    if digest(run.get('market_shared_initialization',{}))!=digest(proof) or plan.get('initialization_contract')!=proof:
        raise RuntimeError('shared initialization provenance mismatch')
    read_initialization(read(prefix+'market-initialization.npz'),initialization_entry(arm,job['seed'],job['smoke']))
    validate_execution_counts(run['training_execution'],arm,job['steps'])
    from rl_stocks.rl.registry.distribution import distribution_spec
    if digest(run.get('action_distribution',{}))!=digest(distribution_spec()):
        raise RuntimeError('market Gaussian distribution metadata required')
    if run['market_initialization']['policy_tensor_sha256']==run['market_final']['policy_tensor_sha256']:
        raise RuntimeError('market training did not change model weights')
    for report_file in ('selected-validation.json','validation.json'):
        validate_execution_report(json.loads(read(prefix+report_file)),arm)
    if (run.get('action_execution')!=modes(arm) or run['training_execution']!=json.loads(read(prefix+'training-execution.json'))):
        raise RuntimeError('actual market execution contract mismatch')
    from rl_stocks.training.market_context import validate_sampling
    checked=validate_sampling(run.get('sampling_diagnostics',{}),arm,job['smoke'],job['steps'])
    if (run.get('dataset')!=recurrent_protocol()['datasets'][arm]
        or frozen['normalizer'].get('clip_abs') is not None
        or run.get('study_protocol_sha256')!=PROTOCOL_SHA256
        or run.get('start_sampling')!=sampling_summary('expanded',job['smoke'])
        or run['sampling_diagnostics']!=json.loads(read(prefix+'training-sampling.json'))
        or digest(frozen['normalizer'])!=recurrent_protocol()['dataset_contracts'][arm]['normalizer_sha256']
        or frozen['names']!=recurrent_protocol()['dataset_contracts'][arm]['names'] or frozen.get('feature_schema')!=recurrent_protocol()['dataset_contracts'][arm]['schema']
        or frozen['data_manifest'].get('included_splits')!=['train','validation']
        or frozen['metadata']!={k:v for k,v in run.items() if k!='bundle_id'}):
        raise RuntimeError('market bundle/data/sampling contract mismatch')

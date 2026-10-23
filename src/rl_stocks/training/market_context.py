"""Fixed compact market-context information ablation and verified mapped initialization."""
import hashlib,json,math
import numpy as np
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.risk.exposure import execution_contract
from rl_stocks.training.recurrent_record import model_record as base_record

PROTOCOL_SHA256='f9045fe85e0e756b3397cb27d4231a82ced8645b1a41c1b20cba093160d7cbad'
ROOT=Path(__file__).resolve().parents[3]


def protocol():
    spec=json.loads((ROOT/'configs/experiments/market_context.json').read_text())
    if digest(spec)!=PROTOCOL_SHA256:raise ValueError('fixed market protocol changed')
    return spec


def width(arm):
    if arm not in protocol()['arms']:raise ValueError('fixed market arm required')
    return 64


def modes(arm):
    width(arm);return execution_contract('unrestricted','unit_magnitude')


def validate_execution_counts(counts,arm,steps):
    width(arm)
    from rl_stocks.training.recurrent_alignment import validate_execution_counts as base
    return base(counts,'posthoc_unit',steps)


def validate_execution_report(results,arm):
    width(arm)
    from rl_stocks.training.recurrent_alignment import validate_execution_report as base
    return base(results,'posthoc_unit')


def identify_arm(metadata):
    arm=metadata.get('market_arm');width(arm)
    if (metadata.get('algorithm')!='recurrent_ppo' or metadata.get('dataset')!=protocol()['datasets'][arm]
            or metadata.get('action_execution')!=modes(arm)):
        raise ValueError('actual market arm/data/execution identity mismatch')
    return arm


def identify_parameters(params,smoke,dataset_id):
    matches=[a for a,d in protocol()['datasets'].items() if dataset_id==d and params==overrides(a,smoke)]
    if len(matches)!=1:raise ValueError('fixed market parameters and arm dataset required')
    return matches[0]


def overrides(arm,smoke=False):
    width(arm)
    from rl_stocks.training.market_initialization import parameters
    return parameters(smoke)


def validate_payload(data,arm):
    width(arm);spec=protocol();contract=spec['dataset_contracts'][arm]
    if (digest(data)!=spec['datasets'][arm] or data['schema']!=contract['schema'] or data['names']!=contract['names']
            or digest(data['normalizer'])!=contract['normalizer_sha256']
            or data['manifest']['included_splits']!=['train','validation']):
        raise ValueError('fixed market dataset/preprocessing identity required')
    if arm=='legacy':
        from rl_stocks.training.diversity import validate_payload as base
        base(data,'expanded')
    else:
        lineage=data['manifest'].get('market_context',{})
        from rl_stocks.data.generation.market_context import validate_frozen_contract
        validate_frozen_contract(dict(data_manifest=data['manifest'],feature_schema=data['schema'],names=data['names'],normalizer=data['normalizer']))
        if (lineage.get('mode')!=arm or lineage.get('source_dataset')!=spec['source_dataset']
                or lineage.get('source_normalizer_sha256')!=spec['original_normalizer_sha256']
                or lineage.get('source_episodes_sha256')!=spec['raw_episodes_sha256']
                or lineage.get('extra_normalizer_fit_rows')!=contract['extra_normalizer_fit_rows']):
            raise ValueError('fixed original source/normalization/fit lineage required')
    return data


def initialization_manifest():
    spec=protocol()['shared_initialization'];path=ROOT/spec['manifest']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['manifest_sha256']:raise ValueError('market initialization manifest mismatch')
    manifest=json.loads(path.read_text())
    if (manifest.get('version')!=1 or manifest.get('preregistration_sha256')!=protocol()['preregistration_sha256']
            or manifest.get('generated_without_fitting') is not True or manifest.get('no_dataset_or_provider_access') is not True):
        raise ValueError('fixed untrained mapping provenance required')
    return manifest


def initialization_entry(arm,seed=11,smoke=False):
    width(arm)
    if type(seed) is not int or seed not in (11,23,37) or type(smoke) is not bool or smoke and seed!=11:
        raise ValueError('fixed market seed/phase required')
    entry=initialization_manifest()['artifacts'][arm+'-'+('smoke' if smoke else 'screen')+'-'+str(seed)]
    if entry['seed']!=seed or entry['smoke'] is not smoke or not (ROOT/entry['file']).resolve().is_relative_to(ROOT/'configs/initializations'):
        raise ValueError('initialization seed/phase/path mismatch')
    return entry


def initialization_spec(arm,seed=11,smoke=False):
    e=initialization_entry(arm,seed,smoke)
    return dict(version=1,artifact_sha256=e['sha256'],policy_tensor_sha256=e['policy_tensor_sha256'],rng_sha256=e['rng_sha256'],
        torch_version=e['torch_version'],generated_without_fitting=True,seed=seed,smoke=smoke,artifact_file=e['file'])


def validate_initialization_assets(arm,seed=11,smoke=False):
    import torch
    from rl_stocks.training.shared_initialization import read_initialization
    e=initialization_entry(arm,seed,smoke)
    if torch.__version__.split('+')[0]!=e['torch_version']:raise ValueError('market Torch version mismatch')
    manifest=initialization_manifest()
    from rl_stocks.training.market_initialization import validate_mapping
    entries={a:initialization_entry(a,seed,smoke) for a in protocol()['arms']}
    arrays={};rngs={}
    for a,entry in entries.items():arrays[a],rngs[a]=read_initialization((ROOT/entry['file']).read_bytes(),entry)
    validate_mapping(arrays['legacy'],arrays['full'],manifest['mapping_proofs'][str(seed)]['new_columns_sha256'])
    if entries['masked']!=entries['full'] or any(not np.array_equal(rngs['legacy'],r) for r in rngs.values()):
        raise ValueError('mapped full/masked tensors and original per-seed RNG required')
    return initialization_spec(arm,seed,smoke)


def apply_initialization(model,arm,seed=11,smoke=False):
    from rl_stocks.training.shared_initialization import load_initialization
    e=initialization_entry(arm,seed,smoke);proof=initialization_spec(arm,seed,smoke)
    actual=load_initialization(model,ROOT/e['file'],e)
    if any(actual[k]!=proof[k] for k in actual):raise ValueError('market loaded initialization proof mismatch')
    model._research_initialization=proof
    return proof


def model_record(model):
    result=base_record(model)
    opt=model.policy.optimizer
    result['coupled_adam']=dict(version=1,optimizer_class=type(opt).__name__,
        defaults_decoupled_weight_decay=opt.defaults.get('decoupled_weight_decay'),
        group_decoupled_weight_decay=[g.get('decoupled_weight_decay') for g in opt.param_groups])
    return result


def validate_record(record,arm,smoke=False,initial=False,selected_steps=None,seed=11):
    from rl_stocks.training.recurrent_record import validate_record as base
    spec=protocol()
    coupling=record.get('coupled_adam',{})
    if (coupling!=dict(version=1,optimizer_class='Adam',defaults_decoupled_weight_decay=False,group_decoupled_weight_decay=[False])
        or type(coupling.get('version')) is not int or coupling.get('defaults_decoupled_weight_decay') is not False
        or any(v is not False for v in coupling.get('group_decoupled_weight_decay',[]))):
        raise ValueError('actual coupled Adam semantics required')
    return base(record,overrides(arm,smoke),spec['observation_sizes'][arm],spec['smoke_steps' if smoke else 'steps_per_cell'],
        initialization_spec(arm,seed,smoke),initial=initial,selected_steps=selected_steps,weight_decay=0.)


def validate_initial_records(records,smoke=False,seed=11):
    if set(records)!=set(protocol()['arms']):raise ValueError('all market initial records required')
    for arm,r in records.items():validate_record(r,arm,smoke,initial=True,seed=seed)
    if len({r['shared_initialization']['rng_sha256'] for r in records.values()})!=1:
        raise ValueError('paired initial CPU sampling RNG required')
    if records['masked']!=records['full']:raise ValueError('full/masked initial tensors and records must match')
    validate_initialization_assets('full',seed,smoke)
    return dict(pinned_per_arm_tensors=True,sampling_rng_identical=True,full_masked_tensors_identical=True,common_columns_verified=True)


def runtime_projection(durations):
    spec=protocol();rule=spec['runtime_projection']
    if (not isinstance(durations,dict) or set(durations)!=set(spec['arms']) or any(isinstance(v,bool) or not isinstance(v,(int,float))
            or not math.isfinite(v) or v<=0 for v in durations.values())):
        raise ValueError('positive actual smoke durations for all arms required')
    ratio=max(1.,*(durations[a]/durations['legacy'] for a in ('masked','full')))
    seconds=rule['safety_multiplier']*rule['reference_control_seconds']*ratio+rule['setup_retrieval_shutdown_reserve_seconds']
    return dict(projected_seconds=seconds,smoke_duration_ratio=ratio,
        screen_allocation_allowed=seconds<=rule['maximum_allocation_seconds'],heuristic_not_guarantee=True)


def make_runtime_profile(root=ROOT):
    path=Path(root)/'.runtime/market-smoke-acceptance.json'
    result=json.loads(path.read_text())
    if (result.get('passed') is not True or result.get('protocol_sha256')!=PROTOCOL_SHA256
            or result.get('cleanup_verified') is not True or result.get('actual_sampling_equal') is not True
            or any(result.get('initialization_checks',{}).get(k) is not True for k in ('sampling_rng_identical','full_masked_tensors_identical','common_columns_verified'))
            or set(result.get('records',{}))!=set(protocol()['arms'])
            or result.get('records',{}).get('masked',{}).get('market_mask_rejection') is not True
            or any(r.get('actual_saved_models_restored') is not True for r in result['records'].values())):
        raise ValueError('actual accepted owned-cleaned market smokes required')
    durations={a:r['duration_seconds'] for a,r in result['records'].items()}
    profile=dict(version=1,protocol_sha256=PROTOCOL_SHA256,smoke_acceptance_file='.runtime/market-smoke-acceptance.json',
        smoke_acceptance_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),durations_seconds=durations,
        projection=runtime_projection(durations))
    validate_runtime_profile(profile)
    return profile


def validate_runtime_profile(profile,root=None):
    required={'version','protocol_sha256','smoke_acceptance_file','smoke_acceptance_sha256','durations_seconds','projection'}
    if (not isinstance(profile,dict) or set(profile)!=required or type(profile['version']) is not int or profile['version']!=1
            or profile['protocol_sha256']!=PROTOCOL_SHA256 or profile['smoke_acceptance_file']!='.runtime/market-smoke-acceptance.json'
            or not isinstance(profile['smoke_acceptance_sha256'],str) or len(profile['smoke_acceptance_sha256'])!=64 or any(c not in '0123456789abcdef' for c in profile['smoke_acceptance_sha256'])
            or profile['projection']!=runtime_projection(profile['durations_seconds'])
            or profile['projection']['screen_allocation_allowed'] is not True):
        raise ValueError('accepted under-deadline market runtime profile required')
    if root is not None and profile!=make_runtime_profile(root):raise ValueError('runtime profile differs from actual smoke acceptance')
    return profile


def validate_sampling(counts,arm,smoke,steps):
    width(arm)
    from rl_stocks.training.diversity import validate_counts
    from rl_stocks.environment.sampling.years import validate_year_counts
    result=validate_counts(counts,'expanded',smoke,steps)
    years=validate_year_counts(counts.get('year_counts',{}),counts['starts'],steps)
    if any(int(y)<2010 for field in ['start_years','decision_years','execution_years'] for y,n in years[field].items() if n):
        raise ValueError('full-history market year count required')
    return result

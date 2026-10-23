"""Fixed tanh-action distribution ablation; unchanged raw data and recurrent model."""
import hashlib,json,math
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.risk.exposure import execution_contract
from rl_stocks.training.recurrent_record import model_record as base_record

PROTOCOL_SHA256='2195938b456723b0434dfbc39dba06ce288feec56c669477ddf6de75170d17a0'
ROOT=Path(__file__).resolve().parents[3]


def protocol():
    spec=json.loads((ROOT/'configs/experiments/squashed_actions.json').read_text())
    if digest(spec)!=PROTOCOL_SHA256:raise ValueError('fixed squashed protocol changed')
    return spec


def width(arm):
    if arm not in protocol()['arms']:raise ValueError('fixed squashed arm required')
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
    arm=metadata.get('squashed_arm');width(arm);spec=protocol()
    if (metadata.get('algorithm')!=spec['algorithms'][arm] or metadata.get('dataset')!=spec['dataset']
            or metadata.get('action_execution')!=modes(arm)):
        raise ValueError('actual squashed arm/algorithm/data/execution identity mismatch')
    return arm


def identify_parameters(params,smoke,algorithm):
    matches=[a for a in protocol()['arms'] if algorithm==protocol()['algorithms'][a] and params==overrides(a,smoke)]
    if len(matches)!=1:raise ValueError('fixed squashed parameters and arm algorithm required')
    return matches[0]


def overrides(arm,smoke=False):
    w=width(arm)
    if type(smoke) is not bool:raise ValueError('explicit boolean smoke required')
    spec=protocol();params=dict(spec['parameters']);params.pop('net_arch')
    if smoke:params.update(spec['smoke_parameters'])
    return dict(params,target_kl=None,policy_kwargs=dict(spec['recurrent_parameters'],net_arch=[w,w],lstm_hidden_size=w,optimizer_kwargs=dict(eps=1e-5,weight_decay=0.)))


def validate_payload(data):
    spec=protocol()
    from rl_stocks.training.diversity import validate_payload as base
    base(data,'expanded')
    if (digest(data)!=spec['dataset'] or digest(data['normalizer'])!=spec['normalizer_sha256']
            or data['schema']!=spec['schema'] or digest(data['episodes'])!=spec['raw_episodes_sha256']):
        raise ValueError('unchanged squashed study source required')
    return data


def initialization_entry(arm,seed=11,smoke=False):
    width(arm)
    spec=protocol()['shared_initialization']
    if hashlib.sha256((ROOT/spec['manifest']).read_bytes()).hexdigest()!=spec['manifest_sha256']:
        raise ValueError('squashed initialization manifest mismatch')
    from rl_stocks.training.recurrent_capacity import initialization_entry as shared_entry
    return shared_entry('control',seed,smoke)


def initialization_spec(arm,seed=11,smoke=False):
    e=initialization_entry(arm,seed,smoke)
    return dict(version=1,artifact_sha256=e['sha256'],policy_tensor_sha256=e['policy_tensor_sha256'],rng_sha256=e['rng_sha256'],
        torch_version=e['torch_version'],generated_without_fitting=True,seed=seed,smoke=smoke,artifact_file=e['file'])


def validate_initialization_assets(arm,seed=11,smoke=False):
    import torch
    from rl_stocks.training.shared_initialization import read_initialization
    e=initialization_entry(arm,seed,smoke)
    if torch.__version__.split('+')[0]!=e['torch_version']:raise ValueError('squashed Torch version mismatch')
    read_initialization((ROOT/e['file']).read_bytes(),e)
    return initialization_spec(arm,seed,smoke)


def apply_initialization(model,arm,seed=11,smoke=False):
    from rl_stocks.training.shared_initialization import load_initialization
    e=initialization_entry(arm,seed,smoke);proof=initialization_spec(arm,seed,smoke)
    actual=load_initialization(model,ROOT/e['file'],e)
    if any(actual[k]!=proof[k] for k in actual):raise ValueError('squashed loaded initialization proof mismatch')
    model._research_initialization=proof
    return proof


def model_record(model):
    result=base_record(model)
    opt=model.policy.optimizer
    result['coupled_adam']=dict(version=1,optimizer_class=type(opt).__name__,
        defaults_decoupled_weight_decay=opt.defaults.get('decoupled_weight_decay'),
        group_decoupled_weight_decay=[g.get('decoupled_weight_decay') for g in opt.param_groups])
    from rl_stocks.rl.registry.distribution import model_distribution
    result.update(action_distribution=model_distribution(model),use_sde=model.use_sde,squash_output=model.policy.squash_output)
    return result


def validate_record(record,arm,smoke=False,initial=False,selected_steps=None,seed=11):
    from rl_stocks.training.recurrent_record import validate_record as base
    spec=protocol()
    width(arm)
    bounded=arm=='squashed'
    distribution=({'version':1,'kind':'tanh_squashed_diagonal_gaussian','components':4,'jacobian_epsilon':1e-6}
        if bounded else {'version':1,'kind':'diagonal_gaussian','components':4})
    if (digest(record.get('action_distribution'))!=digest(distribution) or record.get('use_sde') is not False
            or record.get('squash_output') is not bounded):
        raise ValueError('actual squashed action distribution/support contract required')
    coupling=record.get('coupled_adam',{})
    if (coupling!=dict(version=1,optimizer_class='Adam',defaults_decoupled_weight_decay=False,group_decoupled_weight_decay=[False])
        or type(coupling.get('version')) is not int or coupling.get('defaults_decoupled_weight_decay') is not False
        or any(v is not False for v in coupling.get('group_decoupled_weight_decay',[]))):
        raise ValueError('actual coupled Adam semantics required')
    return base(record,overrides(arm,smoke),21,spec['smoke_steps' if smoke else 'steps_per_cell'],
        initialization_spec(arm,seed,smoke),initial=initial,selected_steps=selected_steps,weight_decay=0.,policy_class='SquashedRecurrentPolicy' if bounded else 'RecurrentActorCriticPolicy')


def validate_initial_records(records,smoke=False,seed=11):
    if set(records)!=set(protocol()['arms']):raise ValueError('both squashed initial records required')
    for arm,r in records.items():validate_record(r,arm,smoke,initial=True,seed=seed)
    if len({r['shared_initialization']['rng_sha256'] for r in records.values()})!=1:
        raise ValueError('paired initial CPU sampling RNG required')
    if len({r['policy_tensor_sha256'] for r in records.values()})!=1:raise ValueError('same initial squashed tensors required')
    return dict(pinned_per_arm_tensors=True,sampling_rng_identical=True,cross_arm_tensors_identical=True)


def runtime_projection(durations):
    spec=protocol();rule=spec['runtime_projection']
    if (not isinstance(durations,dict) or set(durations)!=set(spec['arms']) or any(isinstance(v,bool) or not isinstance(v,(int,float))
            or not math.isfinite(v) or v<=0 for v in durations.values())):
        raise ValueError('positive actual smoke durations for both arms required')
    ratio=max(1.,durations['squashed']/durations['control'])
    seconds=rule['safety_multiplier']*rule['reference_control_seconds']*ratio+rule['setup_retrieval_shutdown_reserve_seconds']
    return dict(projected_seconds=seconds,smoke_duration_ratio=ratio,
        screen_allocation_allowed=seconds<=rule['maximum_allocation_seconds'],heuristic_not_guarantee=True)


def make_runtime_profile(root=ROOT):
    path=Path(root)/'.runtime/squashed-smoke-acceptance.json'
    result=json.loads(path.read_text())
    if (result.get('passed') is not True or result.get('protocol_sha256')!=PROTOCOL_SHA256
            or result.get('cleanup_verified') is not True or result.get('actual_sampling_equal') is not True
            or result.get('initialization_checks',{}).get('sampling_rng_identical') is not True
            or set(result.get('records',{}))!=set(protocol()['arms'])
            or any(r.get('actual_saved_models_restored') is not True for r in result['records'].values())):
        raise ValueError('actual accepted owned-cleaned squashed smokes required')
    durations={a:r['duration_seconds'] for a,r in result['records'].items()}
    profile=dict(version=1,protocol_sha256=PROTOCOL_SHA256,smoke_acceptance_file='.runtime/squashed-smoke-acceptance.json',
        smoke_acceptance_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),durations_seconds=durations,
        projection=runtime_projection(durations))
    validate_runtime_profile(profile)
    return profile


def validate_runtime_profile(profile,root=None):
    required={'version','protocol_sha256','smoke_acceptance_file','smoke_acceptance_sha256','durations_seconds','projection'}
    if (not isinstance(profile,dict) or set(profile)!=required or type(profile['version']) is not int or profile['version']!=1
            or profile['protocol_sha256']!=PROTOCOL_SHA256 or profile['smoke_acceptance_file']!='.runtime/squashed-smoke-acceptance.json'
            or not isinstance(profile['smoke_acceptance_sha256'],str) or len(profile['smoke_acceptance_sha256'])!=64 or any(c not in '0123456789abcdef' for c in profile['smoke_acceptance_sha256'])
            or profile['projection']!=runtime_projection(profile['durations_seconds'])
            or profile['projection']['screen_allocation_allowed'] is not True):
        raise ValueError('accepted under-deadline squashed runtime profile required')
    if root is not None and profile!=make_runtime_profile(root):raise ValueError('runtime profile differs from actual smoke acceptance')
    return profile


def validate_sampling(counts,arm,smoke,steps):
    width(arm)
    from rl_stocks.training.diversity import validate_counts
    from rl_stocks.environment.sampling.years import validate_year_counts
    result=validate_counts(counts,'expanded',smoke,steps)
    years=validate_year_counts(counts.get('year_counts',{}),counts['starts'],steps)
    if any(int(y)<2010 for field in ['start_years','decision_years','execution_years'] for y,n in years[field].items() if n):
        raise ValueError('full-history squashed year count required')
    return result

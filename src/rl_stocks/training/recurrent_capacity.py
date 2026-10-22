"""One fixed recurrent width ablation, with no feature or budget sweep."""
import hashlib,json,math
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.risk.exposure import execution_contract
from rl_stocks.training.recurrent_record import model_record

PROTOCOL_SHA256='ce6ea75e0a0dd286a9b4f8688f52aeb14c5ea3655084f33f52140c55f1801f93'
ROOT=Path(__file__).resolve().parents[3]


def protocol():
    spec=json.loads((ROOT/'configs/experiments/recurrent_capacity.json').read_text())
    if digest(spec)!=PROTOCOL_SHA256:raise ValueError('fixed capacity protocol changed')
    return spec


def width(arm):
    if arm not in protocol()['arms']:raise ValueError('fixed capacity arm required')
    return protocol()['widths'][arm]


def modes(arm):
    width(arm);return execution_contract('unrestricted','unit_magnitude')


def overrides(arm,smoke=False):
    w=width(arm)
    if type(smoke) is not bool:raise ValueError('explicit boolean smoke required')
    spec=protocol();params=dict(spec['parameters']);params.pop('net_arch')
    if smoke:params.update(spec['smoke_parameters'])
    return dict(params,target_kl=None,policy_kwargs=dict(spec['recurrent_parameters'],net_arch=[w,w],lstm_hidden_size=w))


def validate_payload(data):
    from rl_stocks.training.diversity import validate_payload as base
    return base(data,'expanded')


def initialization_entry(arm,seed=11,smoke=False):
    width(arm)
    if type(seed) is not int or seed not in (11,23,37) or type(smoke) is not bool or smoke and seed!=11:
        raise ValueError('fixed capacity seed/phase required')
    spec=protocol()['shared_initialization'];path=ROOT/spec['manifest']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['manifest_sha256']:raise ValueError('capacity initialization manifest mismatch')
    manifest=json.loads(path.read_text())
    if manifest['version']!=1 or manifest['generated_without_fitting'] is not True or manifest['no_dataset_or_provider_access'] is not True:
        raise ValueError('untrained initialization provenance required')
    e=manifest['artifacts'][arm+'-'+('smoke' if smoke else 'screen')+'-'+str(seed)]
    if e['seed']!=seed or e['smoke'] is not smoke or not (ROOT/e['file']).resolve().is_relative_to(ROOT/'configs/initializations'):
        raise ValueError('capacity initialization seed/phase/path mismatch')
    return e


def initialization_spec(arm,seed=11,smoke=False):
    e=initialization_entry(arm,seed,smoke)
    return dict(version=1,artifact_sha256=e['sha256'],policy_tensor_sha256=e['policy_tensor_sha256'],rng_sha256=e['rng_sha256'],
        torch_version=e['torch_version'],generated_without_fitting=True,seed=seed,smoke=smoke,artifact_file=e['file'])


def validate_initialization_assets(arm,seed=11,smoke=False):
    import torch
    from rl_stocks.training.shared_initialization import read_initialization
    e=initialization_entry(arm,seed,smoke)
    if torch.__version__.split('+')[0]!=e['torch_version']:raise ValueError('capacity Torch version mismatch')
    read_initialization((ROOT/e['file']).read_bytes(),e)
    return initialization_spec(arm,seed,smoke)


def apply_initialization(model,arm,seed=11,smoke=False):
    from rl_stocks.training.shared_initialization import load_initialization
    e=initialization_entry(arm,seed,smoke);proof=initialization_spec(arm,seed,smoke)
    actual=load_initialization(model,ROOT/e['file'],e)
    if any(actual[k]!=proof[k] for k in actual):raise ValueError('capacity loaded initialization proof mismatch')
    model._research_initialization=proof
    return proof


def validate_record(record,arm,smoke=False,initial=False,selected_steps=None,seed=11):
    from rl_stocks.training.recurrent_record import validate_record as base
    spec=protocol()
    return base(record,overrides(arm,smoke),21,spec['smoke_steps' if smoke else 'steps_per_cell'],
        initialization_spec(arm,seed,smoke),initial=initial,selected_steps=selected_steps)


def validate_initial_records(records,smoke=False,seed=11):
    if set(records)!=set(protocol()['arms']):raise ValueError('both capacity initial records required')
    for arm,r in records.items():validate_record(r,arm,smoke,initial=True,seed=seed)
    if len({r['shared_initialization']['rng_sha256'] for r in records.values()})!=1:
        raise ValueError('paired initial CPU sampling RNG required')
    return dict(pinned_per_arm_tensors=True,sampling_rng_identical=True,cross_width_tensors_identical=False)


def runtime_projection(durations):
    spec=protocol();rule=spec['runtime_projection']
    if (set(durations)!=set(spec['arms']) or any(isinstance(v,bool) or not isinstance(v,(int,float))
            or not math.isfinite(v) or v<=0 for v in durations.values())):
        raise ValueError('positive actual smoke durations for both arms required')
    ratio=max(1.,durations['larger']/durations['control'])
    seconds=rule['safety_multiplier']*rule['reference_control_seconds']*ratio+rule['setup_retrieval_shutdown_reserve_seconds']
    return dict(projected_seconds=seconds,smoke_duration_ratio=ratio,
        screen_allocation_allowed=seconds<=rule['maximum_allocation_seconds'],heuristic_not_guarantee=True)

"""Fixed policy-update safeguard hypothesis, not a cash-family extension."""
import json,math
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.risk.exposure import execution_contract
from rl_stocks.training.recurrent_record import model_record

PROTOCOL_SHA256='ff999f0508ca0eb65a007eb1a7a9dd64deaf331ab1118a9f4ea044ccab6d252c'


def protocol():
    spec=json.loads((Path(__file__).resolve().parents[3]/'configs/experiments/recurrent_guard.json').read_text())
    if digest(spec)!=PROTOCOL_SHA256:raise ValueError('fixed update-guard protocol changed')
    return spec


def target(arm):
    spec=protocol()
    if arm not in spec['arms']:raise ValueError('fixed update-guard arm required')
    return spec['target_kl'][arm]


def modes(arm):
    target(arm)
    return execution_contract('unrestricted','unit_magnitude')


def overrides(arm,smoke=False):
    value=target(arm)
    if type(smoke) is not bool:raise ValueError('explicit boolean smoke required')
    from rl_stocks.training.recurrent_state import overrides as base
    return dict(base('recurrent_masked',smoke),target_kl=value)


def validate_payload(data):
    from rl_stocks.training.diversity import validate_payload as base
    return base(data,'expanded')


def validate_record(record,arm,smoke=False,initial=False,selected_steps=None,seed=11):
    from rl_stocks.training.recurrent_state import validate_record as base
    base(record,'recurrent_masked',smoke,initial,selected_steps)
    from rl_stocks.core.contracts import canonical
    proof=shared_initialization_spec(seed,smoke)
    if canonical(record.get('shared_initialization'))!=canonical(proof):
        raise ValueError('shared initialization proof mismatch')
    if initial and record['policy_tensor_sha256']!=proof['policy_tensor_sha256']:
        raise ValueError('exact shared initial tensors required')
    if (record.get('target_kl')!=target(arm) or record.get('clip_range')!=.2
            or record.get('max_grad_norm')!=.5 or record.get('policy_class')!='RecurrentActorCriticPolicy'):
        raise ValueError('actual policy-update guard settings mismatch')
    o=record['optimizer'];params=overrides(arm,smoke)
    counters={'num_timesteps':record['num_timesteps'],'n_steps':record['n_steps'],
              'batch_size':record['batch_size'],'n_epochs':record['n_epochs'],'target_kl':target(arm)}
    if any(o.get(k)!=v for k,v in counters.items()) or o.get('all_parameters_and_moments_finite') is not True:
        raise ValueError('actual optimizer settings/finite evidence mismatch')
    h=params['policy_kwargs']['lstm_hidden_size'];w=params['policy_kwargs']['net_arch'][0]
    count=2*4*h*(21+h+2)+2*((h*w+w)+(w*w+w))+4*(w+1)+4+(w+1)
    if o.get('optimizer_parameter_count')!=21 or o.get('parameter_count')!=count:
        raise ValueError('actual optimizer parameter ownership mismatch')
    nominal=o.get('sb3_nominal_epoch_counter');expected=record['num_timesteps']//record['n_steps']*record['n_epochs']
    allowed={0} if initial else {expected}
    if selected_steps is not None:allowed.add(max(0,expected-record['n_epochs']))
    if type(nominal) is not int or nominal not in allowed:raise ValueError('nominal epoch evidence mismatch')
    if initial:
        if o.get('optimizer_state_count')!=0 or o.get('adam_step_min') is not None or o.get('adam_step_max') is not None or o.get('per_parameter_adam_steps')!={}:
            raise ValueError('initial optimizer state must be empty')
    else:
        steps=o.get('adam_step_min');upper=nominal*(record['n_steps']//record['batch_size'])
        if (o.get('optimizer_state_count')!=21 or type(steps) is not int or not 0<steps<=upper
                or o.get('adam_step_max')!=steps or len(o.get('per_parameter_adam_steps',{}))!=21
                or any(type(v) is not int or v!=steps for v in o['per_parameter_adam_steps'].values())
                or arm=='control' and steps!=upper):
            raise ValueError('actual Adam update evidence mismatch')
    for k in ('maximum_abs_first_moment','maximum_second_moment'):
        if not math.isfinite(o[k]) or o[k]<0:raise ValueError('finite nonnegative moment evidence required')
    if o.get('parameter_groups')!=[dict(lr=.0003,eps=1e-5,betas=[.9,.999],weight_decay=0)]:
        raise ValueError('unchanged Adam settings required')
    return record


def validate_execution_counts(counts,arm,steps):
    target(arm)
    from rl_stocks.training.recurrent_alignment import validate_execution_counts as base
    return base(counts,'posthoc_unit',steps)


def validate_execution_report(results,arm):
    target(arm)
    from rl_stocks.training.recurrent_alignment import validate_execution_report as base
    return base(results,'posthoc_unit')


def activation(control,candidate):
    """Descriptive final-step reduction; never substitutes for return/risk gates."""
    proof=control.get('shared_initialization')
    if not isinstance(proof,dict):raise ValueError('shared initialization required for activation')
    seed=proof.get('seed')
    validate_record(control,'control',seed=seed);validate_record(candidate,'kl_guard',seed=seed)
    c=control['optimizer']['adam_step_min'];g=candidate['optimizer']['adam_step_min']
    return dict(control_adam_steps=c,candidate_adam_steps=g,guard_activated=g<c)


def identify_arm(metadata):
    matches=[a for a in protocol()['arms'] if metadata.get('algorithm')=='recurrent_ppo'
        and metadata.get('guard_final',{}).get('target_kl','missing')==target(a)
        and metadata.get('action_execution')==modes(a)]
    if len(matches)!=1:raise ValueError('exact update-guard arm identity required')
    return matches[0]


def validate_initial_pairs(records,smoke=False,seed=11):
    if set(records)!=set(protocol()['arms']):raise ValueError('both initial models required')
    normalized=[]
    for arm,r in records.items():
        validate_record(r,arm,smoke,initial=True,seed=seed)
        clean=json.loads(json.dumps(r));clean.pop('target_kl');clean['optimizer'].pop('target_kl')
        normalized.append(clean)
    if normalized[0]!=normalized[1]:raise ValueError('initial networks/settings differ beyond target_kl')
    return True


def screen_verdict(cells):
    from rl_stocks.reporting.history import screen_gate
    verdict=screen_gate(cells,protocol())
    activated=activation(cells['control']['final_model_record'],cells['kl_guard']['final_model_record'])
    verdict['performance_gates_passed']=verdict['passed']
    verdict['passed']=verdict['passed'] and activated['guard_activated']
    if not verdict['passed']:verdict['promoted_candidate']=None
    verdict['activation']=activated
    return verdict


def initialization_entry(seed=11,smoke=False):
    import hashlib
    if type(seed) is not int or seed not in (11,23,37) or type(smoke) is not bool or smoke and seed!=11:
        raise ValueError('fixed initialization seed/size required')
    root=Path(__file__).resolve().parents[3];spec=protocol()['shared_initialization']
    path=root/spec['manifest']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['manifest_sha256']:
        raise ValueError('initialization manifest checksum mismatch')
    manifest=json.loads(path.read_text())
    if manifest['version']!=1 or manifest['generated_without_fitting'] is not True or manifest['no_dataset_or_provider_access'] is not True:
        raise ValueError('untrained initialization provenance required')
    entry=manifest['artifacts'][('smoke' if smoke else 'screen')+'-'+str(seed)]
    if entry['seed']!=seed or entry['smoke'] is not smoke or not (root/entry['file']).resolve().is_relative_to(root/'configs/initializations'):
        raise ValueError('initialization seed/size/path mismatch')
    return entry


def initialization_path(seed=11,smoke=False):
    return Path(__file__).resolve().parents[3]/initialization_entry(seed,smoke)['file']


def shared_initialization_spec(seed=11,smoke=False):
    e=initialization_entry(seed,smoke)
    return dict(version=1,artifact_sha256=e['sha256'],policy_tensor_sha256=e['policy_tensor_sha256'],
        rng_sha256=e['rng_sha256'],torch_version=e['torch_version'],generated_without_fitting=True,
        seed=seed,smoke=smoke,artifact_file=e['file'])


def validate_initialization_assets(seed=11,smoke=False):
    import torch
    from rl_stocks.training.shared_initialization import read_initialization
    e=initialization_entry(seed,smoke)
    if torch.__version__.split('+')[0]!=e['torch_version']:raise ValueError('shared initialization Torch version mismatch')
    read_initialization(initialization_path(seed,smoke).read_bytes(),e)
    return shared_initialization_spec(seed,smoke)


def apply_shared_initialization(model,seed=11,smoke=False):
    from rl_stocks.training.shared_initialization import load_initialization
    actual=load_initialization(model,initialization_path(seed,smoke),initialization_entry(seed,smoke))
    proof=shared_initialization_spec(seed,smoke)
    if any(actual[k]!=proof[k] for k in actual):raise ValueError('loaded shared initialization proof mismatch')
    model._research_initialization=proof
    return proof

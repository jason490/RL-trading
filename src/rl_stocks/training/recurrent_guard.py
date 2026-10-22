"""Fixed policy-update safeguard hypothesis, not a cash-family extension."""
import json,math
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.risk.exposure import execution_contract

PROTOCOL_SHA256='b67fc7e4be4331b13855354e7daf6c1a4512aa2c5ec45d970ec5cf010ba17cfc'


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


def model_record(model):
    from rl_stocks.training.recurrent_state import model_record as base
    from rl_stocks.reporting.optimizer import optimizer_record
    return dict(base(model),target_kl=model.target_kl,clip_range=float(model.clip_range(1.)),
                max_grad_norm=model.max_grad_norm,policy_class=type(model.policy).__name__,
                optimizer=json.loads(json.dumps(optimizer_record(model))))


def validate_record(record,arm,smoke=False,initial=False,selected_steps=None):
    from rl_stocks.training.recurrent_state import validate_record as base
    base(record,'recurrent_masked',smoke,initial,selected_steps)
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
    validate_record(control,'control');validate_record(candidate,'kl_guard')
    c=control['optimizer']['adam_step_min'];g=candidate['optimizer']['adam_step_min']
    return dict(control_adam_steps=c,candidate_adam_steps=g,guard_activated=g<c)

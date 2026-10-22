"""Fixed cash-availability hypothesis; builders never allocate or fit."""
import json
from pathlib import Path
from rl_stocks.core.contracts import digest

PROTOCOL_SHA256 = '109ccfb4e68b8d83dc62102967dc949ab7cec2df76b9cf801be105eea6130cc0'


def protocol():
    spec=json.loads((Path(__file__).resolve().parents[3]/'configs/experiments/recurrent_cash.json').read_text())
    if digest(spec)!=PROTOCOL_SHA256:
        raise ValueError('cash-choice protocol changed')
    return spec


def arm_spec(arm):
    spec=protocol()
    if arm not in spec['arms']:
        raise ValueError('fixed cash-choice arm required')
    return dict(spec['arm_contracts'][arm])


def overrides(arm, smoke=False):
    contract=arm_spec(arm)
    if type(smoke) is not bool:
        raise ValueError('explicit boolean smoke required')
    spec=protocol()
    params=dict(spec['parameters']);params.pop('net_arch')
    policy=dict(net_arch=list(spec['parameters']['net_arch']),**spec['recurrent_parameters'])
    if smoke:
        for key in ('n_steps','batch_size','n_epochs'):
            params[key]=spec['smoke_parameters'][key]
        policy.update(net_arch=list(spec['smoke_parameters']['net_arch']),
                      lstm_hidden_size=spec['smoke_parameters']['lstm_hidden_size'])
    if contract['distribution']=='direction_protection':
        policy['cash_enabled']=contract['cash_enabled']
    return dict(params,policy_kwargs=policy)


def modes(arm):
    from rl_stocks.risk.exposure import execution_contract
    c=arm_spec(arm)
    return execution_contract(c['training'],c['evaluation'])


def validate_payload(data):
    from rl_stocks.training.diversity import validate_payload as validate
    return validate(data,'expanded')


def model_record(model):
    from rl_stocks.training.recurrent_state import model_record as base
    from rl_stocks.rl.registry.distribution import model_distribution
    return dict(base(model),action_distribution=model_distribution(model),
                parameter_count=sum(p.numel() for p in model.policy.parameters()),
                optimizer_updates=model._n_updates,optimizer_parameter_states=len(model.policy.optimizer.state),
                policy_class=type(model.policy).__name__)


def validate_record(record,arm,smoke=False,initial=False,selected_steps=None):
    from rl_stocks.training.recurrent_state import validate_record as base
    from rl_stocks.rl.registry.distribution import distribution_spec
    contract=arm_spec(arm);params=overrides(arm,smoke)['policy_kwargs']
    base(record,'recurrent_masked',smoke,initial,selected_steps)
    hybrid=contract['distribution']=='direction_protection'
    h=params['lstm_hidden_size'];w=params['net_arch'][0];outputs=6 if hybrid else 4
    # Two separate LSTMs, actor/critic two-layer heads, action/value heads/std.
    count=2*4*h*(21+h+2)+2*((h*w+w)+(w*w+w))+outputs*(w+1)+(3 if hybrid else 4)+(w+1)
    expected=distribution_spec(contract['cash_enabled'])
    from rl_stocks.core.contracts import canonical
    if (canonical(record.get('action_distribution'))!=canonical(expected) or record.get('parameter_count')!=count
            or record.get('policy_class')!=('HybridRecurrentPolicy' if hybrid else 'RecurrentActorCriticPolicy')):
        raise ValueError('actual cash-choice policy identity mismatch')
    updates=record.get('optimizer_updates');states=record.get('optimizer_parameter_states')
    epochs=overrides(arm,smoke)['n_epochs'];rollout=overrides(arm,smoke)['n_steps']
    allowed={0} if initial else {record['num_timesteps']//rollout*epochs}
    if selected_steps is not None:allowed.add(max(0,record['num_timesteps']//rollout-1)*epochs)
    if (type(updates) is not int or updates not in allowed or type(states) is not int
            or states<0 or initial and states!=0 or not initial and states<=0):
        raise ValueError('actual cash-choice optimizer update evidence mismatch')
    return record


def validate_execution_counts(counts,arm,steps):
    c=arm_spec(arm)
    if c['distribution']=='gaussian':
        from rl_stocks.training.recurrent_alignment import validate_execution_counts as base
        return base(counts,'posthoc_unit' if arm=='posthoc_unit' else 'aligned_unit',steps)
    fields={'transitions','requested_nonzero','processed_nonunit','sizing_changed','protective_changed'}
    if (set(counts)!={'mode',*fields} or counts['mode']!='unrestricted'
            or any(type(counts[k]) is not int or not 0<=counts[k]<=steps for k in fields)
            or counts['transitions']!=steps or any(counts[k] for k in ('processed_nonunit','sizing_changed','protective_changed'))):
        raise ValueError('actual categorical execution counters mismatch')
    if c['cash_enabled']:
        if not 0<counts['requested_nonzero']<steps:raise ValueError('actual cash and directional samples required')
    elif counts['requested_nonzero']!=steps:raise ValueError('binary policy cannot request cash')
    return counts


def validate_execution_report(results,arm):
    from rl_stocks.training.recurrent_alignment import validate_execution_report as base
    c=arm_spec(arm)
    base(results,'posthoc_unit' if c['evaluation']=='unit_magnitude' else 'adaptive')
    if c['distribution']=='direction_protection':
        import math
        support=(-1,0,1) if c['cash_enabled'] else (-1,1)
        for result in results.values():
            for row in result['rows']:
                a=row['raw_action']
                if len(a)!=4 or a[0] not in support or any(not math.isfinite(v) or abs(v)>1 for v in a):
                    raise ValueError('categorical evaluation action support mismatch')
    return results


def identify_arm(metadata):
    from rl_stocks.rl.registry.distribution import distribution_spec
    matches=[]
    for arm in protocol()['arms']:
        c=arm_spec(arm)
        algorithm='hybrid_recurrent_ppo' if c['distribution']=='direction_protection' else 'recurrent_ppo'
        if (metadata.get('algorithm')==algorithm and metadata.get('action_execution')==modes(arm)
                and metadata.get('action_distribution')==distribution_spec(c['cash_enabled'])):
            matches.append(arm)
    if len(matches)!=1:raise ValueError('exact cash-choice arm identity required')
    return matches[0]


def validate_initial_pairs(records):
    if set(records)!=set(protocol()['arms']):raise ValueError('all four initial records required')
    for first,second in [('posthoc_unit','gaussian_unit'),('categorical_binary','categorical_cash')]:
        left={k:v for k,v in records[first].items() if k!='action_distribution'}
        right={k:v for k,v in records[second].items() if k!='action_distribution'}
        if left!=right:raise ValueError('matched within-family initial models differ')
    return True

"""Shared read-only recurrent model, optimizer and initialization evidence."""
import json,math
from rl_stocks.core.contracts import canonical


def model_record(model):
    from rl_stocks.training.recurrent_state import model_record as base
    from rl_stocks.reporting.optimizer import optimizer_record
    return dict(base(model),target_kl=model.target_kl,clip_range=float(model.clip_range(1.)),
                max_grad_norm=model.max_grad_norm,policy_class=type(model.policy).__name__,
                optimizer=json.loads(json.dumps(optimizer_record(model))),
                shared_initialization=json.loads(json.dumps(getattr(model,'_research_initialization',None))))


def validate_record(record,params,observation_size,final_steps,proof,*,initial=False,selected_steps=None,weight_decay=0.,policy_class='RecurrentActorCriticPolicy'):
    """Validate fixed Gaussian recurrent architecture, checkpoint and Adam evidence."""
    policy=params['policy_kwargs'];obs=observation_size
    fixed=dict(algorithm='RecurrentPPO',observation_size=obs,action_size=4,
        **{k:params[k] for k in ['n_steps','batch_size','n_epochs','gamma','gae_lambda','ent_coef','learning_rate']},
        net_arch=policy['net_arch'],actor_lstm=dict(hidden_size=policy['lstm_hidden_size'],num_layers=1),critic_lstm=True,
        shared_lstm=False,target_kl=None,clip_range=.2,max_grad_norm=.5,policy_class=policy_class)
    if any(record.get(k)!=v for k,v in fixed.items()):raise ValueError('actual recurrent model configuration mismatch')
    final=final_steps;steps=record.get('num_timesteps')
    if selected_steps is not None and (initial or type(selected_steps) is not int or not 0<selected_steps<=final):
        raise ValueError('valid selected step required')
    expected=0 if initial else final if selected_steps is None else selected_steps
    if type(steps) is not int or steps!=expected:raise ValueError('actual recurrent checkpoint step mismatch')
    if canonical(record.get('shared_initialization'))!=canonical(proof):raise ValueError('initialization proof mismatch')
    h=record.get('policy_tensor_sha256','')
    if len(h)!=64 or any(c not in '0123456789abcdef' for c in h) or initial and h!=proof['policy_tensor_sha256']:
        raise ValueError('actual tensor fingerprint required')
    o=record['optimizer'];width=policy['net_arch'][0];hidden=policy['lstm_hidden_size']
    count=2*4*hidden*(obs+hidden+2)+2*((hidden*width+width)+(width*width+width))+4*(width+1)+4+(width+1)
    if (o['parameter_count']!=count or o['optimizer_parameter_count']!=21 or not o['all_parameters_and_moments_finite']
            or any(o[k]!=record[k] for k in ['num_timesteps','n_steps','batch_size','n_epochs','target_kl'])):
        raise ValueError('actual optimizer shape/settings mismatch')
    nominal=steps//params['n_steps']*params['n_epochs'];allowed={0} if initial else {nominal}
    if selected_steps is not None:allowed.add(max(0,nominal-params['n_epochs']))
    epoch=o['sb3_nominal_epoch_counter']
    if type(epoch) is not int or epoch not in allowed:raise ValueError('nominal epoch counter mismatch')
    adam=epoch*(params['n_steps']//params['batch_size'])
    if initial:
        if o['optimizer_state_count']!=0 or o['adam_step_min'] is not None or o['adam_step_max'] is not None or o['per_parameter_adam_steps']!={}:
            raise ValueError('untrained empty optimizer required')
    elif (o['optimizer_state_count']!=21 or type(o['adam_step_min']) is not int or o['adam_step_min']!=adam
            or o['adam_step_max']!=adam or len(o['per_parameter_adam_steps'])!=21
            or any(type(v) is not int or v!=adam for v in o['per_parameter_adam_steps'].values())):
        raise ValueError('actual Adam steps mismatch')
    if any(not math.isfinite(o[k]) or o[k]<0 for k in ['maximum_abs_first_moment','maximum_second_moment']):
        raise ValueError('finite optimizer moments required')
    if isinstance(weight_decay,bool) or not math.isfinite(weight_decay) or weight_decay<0:raise ValueError('fixed finite weight decay required')
    if o['parameter_groups']!=[dict(lr=.0003,eps=1e-5,betas=[.9,.999],weight_decay=weight_decay)]:raise ValueError('unchanged Adam groups required')
    return record



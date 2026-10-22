"""Read-only PPO/Adam evidence; counters and logged KL are not causal proof."""
import math
from statistics import median
import torch
from rl_stocks.training.curves import latest_points


def _step(value):
    if isinstance(value,torch.Tensor):
        if value.numel()!=1:raise ValueError('scalar Adam step required')
        value=value.item()
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or int(value)!=value:
        raise ValueError('nonnegative integer Adam step required')
    return int(value)


def optimizer_record(model):
    policy=model.policy;optimizer=policy.optimizer
    if type(optimizer) is not torch.optim.Adam:raise ValueError('saved Adam optimizer required')
    parameters={id(p):(name,p) for name,p in policy.named_parameters() if p.requires_grad}
    owned=[p for group in optimizer.param_groups for p in group['params']]
    if len(owned)!=len(parameters) or {id(p) for p in owned}!=set(parameters):
        raise ValueError('optimizer must own each trainable policy parameter exactly once')
    if any(not bool(torch.isfinite(p).all()) for _,p in parameters.values()):
        raise ValueError('nonfinite policy parameter')
    states={};moment_max=second_max=0.
    for p,state in optimizer.state.items():
        if id(p) not in parameters:raise ValueError('foreign optimizer state')
        if not state:continue
        name,_=parameters[id(p)]
        for key in ('exp_avg','exp_avg_sq'):
            moment=state[key]
            if moment.shape!=p.shape or not bool(torch.isfinite(moment).all()):
                raise ValueError('nonfinite or misshaped Adam moment')
        if bool((state['exp_avg_sq']<0).any()):raise ValueError('negative second moment')
        states[name]=_step(state['step'])
        moment_max=max(moment_max,float(state['exp_avg'].detach().abs().max()))
        second_max=max(second_max,float(state['exp_avg_sq'].detach().max()))
    steps=list(states.values())
    counter=_step(model._n_updates)
    return dict(num_timesteps=model.num_timesteps,sb3_nominal_epoch_counter=counter,
        n_steps=model.n_steps,batch_size=model.batch_size,n_epochs=model.n_epochs,
        target_kl=model.target_kl,parameter_count=sum(p.numel() for _,p in parameters.values()),
        optimizer_parameter_count=len(parameters),optimizer_state_count=len(states),
        adam_step_min=min(steps) if steps else None,adam_step_max=max(steps) if steps else None,
        per_parameter_adam_steps=states,all_parameters_and_moments_finite=True,
        maximum_abs_first_moment=moment_max,maximum_second_moment=second_max,
        parameter_groups=[{k:group[k] for k in ('lr','eps','betas','weight_decay')} for group in optimizer.param_groups])


def training_diagnostics(history):
    if not history:raise ValueError('saved training curve required')
    points=latest_points(history);records=[]
    fields=('train/approx_kl','train/clip_fraction','train/entropy_loss','train/std')
    for row in points:
        if type(row['steps']) is not int or row['steps']<=0 or row['phase'] not in ('on_step','post_update'):
            raise ValueError('valid saved checkpoint required')
        training=row['training']
        if any(k not in training or not math.isfinite(training[k]) for k in fields):
            raise ValueError('finite saved PPO diagnostics required')
        if (training['train/approx_kl'] < -1e-6 or not 0<=training['train/clip_fraction']<=1
                or training['train/std']<=0 or not math.isfinite(row['score'])):
            raise ValueError('invalid saved PPO diagnostic range')
        records.append(dict(steps=row['steps'],phase=row['phase'],return_pct=100*row['score'],
            **{k:training[k] for k in fields}))
    ranges={k:dict(min=min(r[k] for r in records),median=median(r[k] for r in records),
                   max=max(r[k] for r in records)) for k in fields}
    return dict(points=records,ranges=ranges,
        interpretation='Sparse last-rollout logs, not all optimizer steps. On-step evaluation precedes its rollout update. Approximate KL is logged from the last epoch; clip fraction and entropy loss average the update minibatches. Joint entropy differs across action distributions and is not directly comparable as direction uncertainty. No causal conclusion.')

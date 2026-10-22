"""Fixed session information hypothesis and checksum-pinned untrained starts."""
import hashlib,json,math
from pathlib import Path
from rl_stocks.core.contracts import digest,canonical
from rl_stocks.risk.exposure import execution_contract
from rl_stocks.training.recurrent_guard import model_record

PROTOCOL_SHA256='b5fdd94552db62142aab2c08b3a052133e51c9983d17099e64c6995a21bdf466'
ROOT=Path(__file__).resolve().parents[3]


def protocol():
    spec=json.loads((ROOT/'configs/experiments/session_context.json').read_text())
    if digest(spec)!=PROTOCOL_SHA256:raise ValueError('fixed session-context protocol changed')
    return spec


def arm_spec(arm):
    spec=protocol()
    if arm not in spec['arms']:raise ValueError('fixed session-context arm required')
    return dict(dataset=spec['datasets'][arm],market_features=spec['market_features'][arm],
        observation_size=spec['observation_sizes'][arm],mode=spec['feature_mode'][arm])


def modes(arm):
    arm_spec(arm);return execution_contract('unrestricted','unit_magnitude')


def overrides(arm,smoke=False):
    arm_spec(arm)
    if type(smoke) is not bool:raise ValueError('explicit boolean smoke required')
    spec=protocol();params=dict(spec['parameters']);net=params.pop('net_arch')
    policy=dict(net_arch=list(net),**spec['recurrent_parameters'])
    if smoke:
        for k in ['n_steps','batch_size','n_epochs']:params[k]=spec['smoke_parameters'][k]
        policy.update(net_arch=list(spec['smoke_parameters']['net_arch']),lstm_hidden_size=spec['smoke_parameters']['lstm_hidden_size'])
    return dict(params,target_kl=None,policy_kwargs=policy)


def validate_payload(data,arm):
    expected=arm_spec(arm);spec=protocol();contract=spec['dataset_contracts'][arm]
    if (digest(data)!=expected['dataset'] or data['schema']!=contract['schema'] or data['names']!=contract['names']
            or digest(data['normalizer'])!=contract['normalizer_sha256']
            or data['manifest']['included_splits']!=['train','validation']
            or data['manifest']['purge_sessions']!=20 or data['manifest']['unverified_override'] is not False
            or len([e for e in data['episodes'].values() if e['split']=='train'])!=111
            or len([e for e in data['episodes'].values() if e['split']=='validation'])!=26
            or any(e['split'] not in ('train','validation') or any(b['session']>('2020-12-31' if e['split']=='train' else '2023-12-31') for b in e['bars']) for e in data['episodes'].values())):
        raise ValueError('fixed session dataset/preprocessing/panel required')
    if arm!='legacy' and data['manifest']['session_context']['mode']!=expected['mode']:
        raise ValueError('session information mode mismatch')
    return data


def initialization_entry(arm,seed=11,smoke=False):
    arm_spec(arm)
    if type(seed) is not int or seed not in (11,23,37) or type(smoke) is not bool or smoke and seed!=11:
        raise ValueError('fixed initialization seed/size required')
    spec=protocol()['shared_initialization'];path=ROOT/spec['manifest']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['manifest_sha256']:raise ValueError('initialization manifest mismatch')
    manifest=json.loads(path.read_text())
    if manifest['version']!=1 or manifest['generated_without_fitting'] is not True or manifest['no_dataset_or_provider_access'] is not True:
        raise ValueError('untrained initialization provenance required')
    key=('legacy' if arm=='legacy' else 'augmented')+'-'+('smoke' if smoke else 'screen')+'-'+str(seed)
    e=manifest['artifacts'][key]
    if e['seed']!=seed or e['smoke'] is not smoke or not (ROOT/e['file']).resolve().is_relative_to(ROOT/'configs/initializations'):
        raise ValueError('initialization seed/size/path mismatch')
    return e


def initialization_spec(arm,seed=11,smoke=False):
    e=initialization_entry(arm,seed,smoke)
    return dict(version=1,artifact_sha256=e['sha256'],policy_tensor_sha256=e['policy_tensor_sha256'],
        rng_sha256=e['rng_sha256'],torch_version=e['torch_version'],generated_without_fitting=True,
        seed=seed,smoke=smoke,artifact_file=e['file'])


def validate_initialization_assets(arm,seed=11,smoke=False):
    import torch
    from rl_stocks.training.shared_initialization import read_initialization
    e=initialization_entry(arm,seed,smoke)
    if torch.__version__.split('+')[0]!=e['torch_version']:raise ValueError('initialization Torch version mismatch')
    read_initialization((ROOT/e['file']).read_bytes(),e)
    return initialization_spec(arm,seed,smoke)


def apply_initialization(model,arm,seed=11,smoke=False):
    from rl_stocks.training.shared_initialization import load_initialization
    e=initialization_entry(arm,seed,smoke);actual=load_initialization(model,ROOT/e['file'],e)
    proof=initialization_spec(arm,seed,smoke)
    if any(actual[k]!=proof[k] for k in actual):raise ValueError('loaded initialization proof mismatch')
    model._research_initialization=proof
    return proof


def validate_record(record,arm,smoke=False,initial=False,selected_steps=None,seed=11):
    params=overrides(arm,smoke);spec=protocol();policy=params['policy_kwargs'];obs=arm_spec(arm)['observation_size']
    fixed=dict(algorithm='RecurrentPPO',observation_size=obs,action_size=4,
        **{k:params[k] for k in ['n_steps','batch_size','n_epochs','gamma','gae_lambda','ent_coef','learning_rate']},
        net_arch=policy['net_arch'],actor_lstm=dict(hidden_size=policy['lstm_hidden_size'],num_layers=1),critic_lstm=True,
        shared_lstm=False,target_kl=None,clip_range=.2,max_grad_norm=.5,policy_class='RecurrentActorCriticPolicy')
    if any(record.get(k)!=v for k,v in fixed.items()):raise ValueError('actual session model configuration mismatch')
    final=spec['smoke_steps' if smoke else 'steps_per_cell'];steps=record.get('num_timesteps')
    if selected_steps is not None and (initial or type(selected_steps) is not int or not 0<selected_steps<=final):
        raise ValueError('valid selected step required')
    expected=0 if initial else final if selected_steps is None else selected_steps
    if type(steps) is not int or steps!=expected:raise ValueError('actual session checkpoint step mismatch')
    proof=initialization_spec(arm,seed,smoke)
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
    if o['parameter_groups']!=[dict(lr=.0003,eps=1e-5,betas=[.9,.999],weight_decay=0)]:raise ValueError('unchanged Adam groups required')
    return record


def validate_initial_pairs(records,smoke=False,seed=11):
    if set(records)!=set(protocol()['arms']):raise ValueError('all three initial records required')
    for arm,r in records.items():validate_record(r,arm,smoke,initial=True,seed=seed)
    if records['masked']!=records['full']:raise ValueError('masked/full initial tensors/settings must match exactly')
    return True

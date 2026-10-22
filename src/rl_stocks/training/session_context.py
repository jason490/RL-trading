"""Fixed session information hypothesis and checksum-pinned untrained starts."""
import hashlib,json
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.risk.exposure import execution_contract
from rl_stocks.training.recurrent_record import model_record

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


def validate_execution_counts(counts,arm,steps):
    arm_spec(arm)
    from rl_stocks.training.recurrent_alignment import validate_execution_counts as base
    return base(counts,'posthoc_unit',steps)


def validate_execution_report(results,arm):
    arm_spec(arm)
    from rl_stocks.training.recurrent_alignment import validate_execution_report as base
    return base(results,'posthoc_unit')


def identify_arm(metadata):
    arm=metadata.get('session_context_arm');arm_spec(arm)
    if (metadata.get('algorithm')!='recurrent_ppo' or metadata.get('dataset')!=arm_spec(arm)['dataset']
            or metadata.get('action_execution')!=modes(arm)):
        raise ValueError('actual session arm/data/execution identity mismatch')
    return arm


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
    from rl_stocks.training.recurrent_record import validate_record as validate
    return validate(record,overrides(arm,smoke),arm_spec(arm)['observation_size'],
        protocol()['smoke_steps' if smoke else 'steps_per_cell'],initialization_spec(arm,seed,smoke),
        initial=initial,selected_steps=selected_steps)


def validate_initial_pairs(records,smoke=False,seed=11):
    if set(records)!=set(protocol()['arms']):raise ValueError('all three initial records required')
    for arm,r in records.items():validate_record(r,arm,smoke,initial=True,seed=seed)
    if records['masked']!=records['full']:raise ValueError('masked/full initial tensors/settings must match exactly')
    return True

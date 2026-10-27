"""Bounded allowlisted profile archive; physical optimizer and report acceptance."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
import torch

from rl_stocks.core.contracts import canonical,digest
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.report_contract import validate_development_report
from rl_stocks.runtime.colab.jobs import validate_plan
from rl_stocks.training.recurrent_comparison import protocol,profile_admission
from rl_stocks.training.recurrent_comparison_profile import validate_panel
from rl_stocks.training.recurrent_comparison_starts import lineage
from rl_stocks.training.recurrent_sac_checkpoint import load_checkpoint
from rl_stocks.training.sac_update_worker import preserved_rng

ROOT=Path(__file__).resolve().parents[4]
FILES={'profile.json','partial-profile.json','projection.json','provenance.json',
       'collected-replay.npz','candidate-updates.json','native_sac.zip','recurrent_ppo_reference.zip',
       'gru_sac/state.pt','gru_sac/manifest.json'}
FILES|={f'{arm}-validation.{suffix}' for arm in protocol()['arms'] for suffix in ('json','html')}


def optimizer_steps(optimizer,expected):
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    if len(optimizer.state)!=len(parameters):
        raise ValueError('complete physical Adam state required')
    for parameter in parameters:
        state=optimizer.state[parameter]
        if (set(state)!={'step','exp_avg','exp_avg_sq'} or state['step'].numel()!=1
                or state['step'].item()!=expected
                or any(state[key].shape!=parameter.shape or not bool(torch.isfinite(state[key]).all())
                       for key in ('exp_avg','exp_avg_sq'))):
            raise ValueError('actual finite optimizer moments/counts mismatch')


def verify_directory(folder,completion,*,dataset,starts):
    folder=Path(folder)
    validate_panel(dataset)
    def read(name):
        return json.loads((folder/name).read_text())
    timing=read('profile.json')
    projection=profile_admission(timing)
    if canonical(projection)!=canonical(read('projection.json')) or read('partial-profile.json')!=timing:
        raise ValueError('saved timing/projection identities mismatch')
    provenance=read('provenance.json')
    if (provenance['source_sha256']!=completion['source_hash']
            or provenance['repository']!=completion['source_git']
            or provenance['initialization_manifest_sha256']!=hashlib.sha256((Path(starts)/'manifest.json').read_bytes()).hexdigest()
            or provenance['backend']['selected_device']!='cpu'
            or provenance['performance_interpretation_allowed'] is not False):
        raise ValueError('physical profile source/start/backend lineage mismatch')
    with np.load(folder/'collected-replay.npz',allow_pickle=False) as replay:
        shapes=dict(observations=(512,21),next_observations=(512,21),actions=(512,4),
                    rewards=(512,),terminated=(512,),truncated=(512,))
        if set(replay.files)!=set(shapes) or any(replay[k].shape!=shape or not np.isfinite(replay[k]).all()
                                               for k,shape in shapes.items()):
            raise ValueError('complete finite collected training replay required')
        if (np.abs(replay['actions'])>1).any():
            raise ValueError('bounded recorded actions required')
    records=read('candidate-updates.json')
    if (len(records)!=64 or [r['transitions'] for r in records]!=list(range(71,513,7))
            or any(r['diagnostics']['update']!=i for i,r in enumerate(records,1))):
        raise ValueError('fixed actual candidate update schedule required')
    with preserved_rng():
        candidate,_=load_checkpoint(folder/'gru_sac',expected_lineage=lineage())
        if candidate.updates!=64 or candidate.max_updates!=64:
            raise ValueError('bounded actual GRU profile counts required')
        native=SAC.load(folder/'native_sac.zip',device='cpu')
        ppo=RecurrentPPO.load(folder/'recurrent_ppo_reference.zip',device='cpu')
        if native._n_updates!=64 or ppo._n_updates!=30 or ppo.num_timesteps!=3000:
            raise ValueError('actual native profile counts mismatch')
        for optimizer in (native.actor.optimizer,native.critic.optimizer,native.ent_coef_optimizer):
            optimizer_steps(optimizer,64)
        optimizer_steps(ppo.policy.optimizer,600)
        for arm,model in (('native_sac',native),('recurrent_ppo_reference',ppo)):
            expected=protocol()['arm_contracts'][arm]
            for key,value in expected.items():
                if key in ('algorithm','observation_profile') or key.startswith('expected_'):
                    continue
                actual=getattr(model,key)
                if key=='train_freq':
                    actual=actual.frequency
                if canonical(actual)!=canonical(value):
                    raise ValueError('saved native profile configuration mismatch')
            if model.observation_space.shape!=(21,) or model.action_space.shape!=(4,) or any(
                    not bool(torch.isfinite(p).all()) for p in model.policy.parameters()):
                raise ValueError('finite actual native geometry required')
        for arm in protocol()['arms']:
            validate_development_report(read(arm+'-validation.json'),dataset,SimulationConfig())
    return dict(saved_artifacts_verified=True,profile_projection_passed=projection['profile_projection_passed'],
                performance_interpretation_allowed=False)


def validate_profile_archive(path,plan,dataset_id):
    validate_plan(plan)
    if dataset_id!=protocol()['dataset']:
        raise RuntimeError('fixed profile dataset required')
    required={'artifacts/completion.json'}|{'artifacts/comparison-profile/'+name for name in FILES}
    directories={'artifacts','artifacts/comparison-profile','artifacts/comparison-profile/gru_sac'}
    with tarfile.open(path) as archive:
        members=archive.getmembers()
        files={m.name for m in members if m.isfile()}
        if (len({m.name for m in members})!=len(members) or not required<=files
                or files-required-{'artifacts/bootstrap-log.txt'}
                or any(not m.isfile() and (not m.isdir() or m.name not in directories) for m in members)
                or sum(m.size for m in members)>128*1024*1024):
            raise RuntimeError('complete bounded allowlisted profile archive required')
        with tempfile.TemporaryDirectory(prefix='comparison-profile-verify-') as temporary:
            archive.extractall(temporary,filter='data')
            root=Path(temporary)/'artifacts'
            completion=json.loads((root/'completion.json').read_text())
            if (canonical(completion['plan'])!=canonical(plan) or completion['dataset']!=dataset_id
                    or completion['gpu'] is not None or completion['python'][:2]!=[3,13]
                    or completion['source_git'].get('dirty')!=''
                    or len(completion['source_git'].get('revision',''))!=40
                    or len(completion.get('source_hash',''))!=64):
                raise RuntimeError('actual clean Colab CPU completion required')
            dataset=json.loads((ROOT/'data/datasets'/dataset_id/'dataset.json').read_text())
            proof=verify_directory(root/'comparison-profile',completion,dataset=dataset,
                                   starts=ROOT/'.runtime/recurrent-comparison-starts-v1')
            return [dict(algorithm='recurrent_comparison_profile',seed=11,actual_steps=512,
                         bundle_id=digest(proof),status='complete',**proof)]

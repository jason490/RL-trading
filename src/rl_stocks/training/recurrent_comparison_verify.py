"""Read-only full-development artifact verification; never allocates or trains."""
import csv
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
import torch

from rl_stocks.core.contracts import canonical,digest
from rl_stocks.evaluation.recorded_economics import verify_report
from rl_stocks.evaluation.runner import selection_score
from rl_stocks.reporting.recurrent_sac_comparison import phase_summary
from rl_stocks.training.recurrent_comparison import protocol,PROTOCOL_SHA256,candidate_config,validate_native_parameters
from rl_stocks.training.curves import curve_record
from rl_stocks.training.recurrent_comparison_profile import validate_panel
from rl_stocks.training.recurrent_comparison_starts import accept,lineage as starts_lineage
from rl_stocks.training.recurrent_comparison_worker import expected_counts
from rl_stocks.training.recurrent_sac_checkpoint import load_checkpoint
from rl_stocks.training.sac_update_worker import preserved_rng
from rl_stocks.training.shared_initialization import read_initialization


ROOT_FILES={'comparison.json','run.json','learning-curve.jsonl','learning-curve.csv','learning-curve.html',
            'selected.json','final.json','selected-validation.json','final-validation.json',
            'selected-validation.html','final-validation.html','training-execution.json','training-sampling.json'}


def update_count(arm,steps):
    return max(0,steps-1000) if arm=='gru_sac' else expected_counts(arm,steps)[0]


def verify_history(history,selected,final,arm):
    if (len(history)!=10 or [r['steps'] for r in history]!=list(range(10000,100001,10000))
            or any(r['phase']!='post_update' or type(r['score']) not in (int,float)
                   or not math.isfinite(r['score']) or r['learner_updates']!=update_count(arm,r['steps'])
                   or r['checkpoint']!=f"checkpoint-{r['steps']}" for r in history)):
        raise ValueError('ten fixed finite post-update checkpoint records required')
    for row in history:
        diagnostics=row.get('training',{})
        count=row['learner_updates']
        if arm=='gru_sac':
            valid=(diagnostics.get('update')==count
                   and diagnostics.get('completed_steps')==dict.fromkeys(('actor','critic','entropy'),count)
                   and diagnostics.get('order')==['entropy','critic','actor','targets'])
        else:
            valid=(diagnostics.get('native_library_counter')==count
                   and diagnostics.get('optimizer_steps')==expected_counts(arm,row['steps'])[1]
                   and diagnostics.get('counter_kind')==('gradient_updates' if arm=='native_sac' else 'ppo_epochs'))
        if not valid:
            raise ValueError('curve diagnostics must distinguish actual optimizer steps and library counters')
    best=max(history,key=lambda row:row['score'])  # stable earliest tie, never final-only selection
    for pointer,row in ((selected,best),(final,history[-1])):
        expected=dict(steps=row['steps'],score=row['score'],updates=row['learner_updates'],checkpoint=row['checkpoint'])
        if canonical(pointer)!=canonical(expected):
            raise ValueError('strict earliest-best selected and separate final pointers required')
    return best


def check_moments(optimizer,expected):
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    if expected==0:
        if optimizer.state:
            raise ValueError('initial optimizer must be empty')
        return
    if len(optimizer.state)!=len(parameters):
        raise ValueError('complete actual Adam state required')
    for p in parameters:
        state=optimizer.state[p]
        if (set(state)!={'step','exp_avg','exp_avg_sq'} or state['step'].numel()!=1
                or state['step'].item()!=expected
                or any(state[k].shape!=p.shape or not bool(torch.isfinite(state[k]).all())
                       for k in ('exp_avg','exp_avg_sq'))):
            raise ValueError('actual finite Adam moments and steps mismatch')


def verify_native_checkpoint(folder,*,arm,seed,steps,lineage):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    updates,counts=expected_counts(arm,steps)
    raw=(folder/'model.zip').read_bytes()
    if len(raw)>32*1024*1024:
        raise ValueError('bounded native checkpoint bytes required')
    expected=dict(schema='recurrent-comparison-native-checkpoint-v1',arm=arm,seed=seed,
                  transitions=steps,library_updates=updates,optimizer_steps=counts,lineage=lineage,
                  sha256=hashlib.sha256(raw).hexdigest())
    if canonical(manifest)!=canonical(expected):
        raise ValueError('native checkpoint checksum/identity/counts mismatch')
    with preserved_rng():
        model=(SAC if arm=='native_sac' else RecurrentPPO).load(BytesIO(raw),device='cpu')
    if model.num_timesteps!=steps or model._n_updates!=updates or model.seed!=seed or 'train' in model.__dict__:
        raise ValueError('actual native checkpoint counters/seed/hook mismatch')
    validate_native_parameters(model,arm)
    if model.observation_space.shape!=(21,) or model.action_space.shape!=(4,) or any(
            not bool(torch.isfinite(p).all()) for p in model.policy.parameters()):
        raise ValueError('finite native21-input/four-action geometry required')
    if arm=='native_sac' and (not bool(torch.isfinite(model.log_ent_coef).all())
            or steps==0 and model.log_ent_coef.item()!=0):
        raise ValueError('finite native entropy and exact initial alpha required')
    optimizers=(dict(actor=model.actor.optimizer,critic=model.critic.optimizer,entropy=model.ent_coef_optimizer)
                if arm=='native_sac' else dict(policy=model.policy.optimizer))
    for role,opt in optimizers.items():
        check_moments(opt,counts[role])
    return model


def verify_cell(folder,dataset,starts,*,expected_arm,expected_seed,expected_source_hash,expected_git_revision):
    """Expected source/role/seed must come from the externally admitted completion."""
    spec=protocol()
    if (expected_arm not in spec['arms'] or type(expected_seed) is not int
            or expected_seed not in spec['confirmation_seeds']
            or type(expected_source_hash) is not str or len(expected_source_hash)!=64
            or type(expected_git_revision) is not str or len(expected_git_revision)!=40):
        raise ValueError('externally bound complete run identity required')
    folder=Path(folder)
    validate_panel(dataset)
    starts=Path(starts)
    accept(starts)
    initializations=json.loads((starts/'manifest.json').read_text())
    arm,seed=expected_arm,expected_seed
    def read(name):
        return json.loads((folder/name).read_text())
    meta=read('comparison.json')
    schema=spec['arm_contracts'][arm]['observation_profile'].replace('_','-')+'-v1-normalized21'
    lineage=dict(source=expected_source_hash,dataset=spec['dataset'],observation_schema=schema)
    if (meta['protocol_sha256']!=PROTOCOL_SHA256 or meta['arm']!=arm or meta['seed']!=seed
            or meta['lineage']!=lineage or meta['source_git']!=dict(revision=expected_git_revision,dirty='')
            or meta['initialization_manifest_sha256']!=hashlib.sha256((starts/'manifest.json').read_bytes()).hexdigest()
            or meta['python'][:2]!=[3,13] or meta['backend']['selected_device']!='cpu'
            or meta['checkpoint_phase']!='post_update' or meta['transitions']!=100000
            or meta['evaluation_every']!=10000 or meta['retired_test_access'] is not False):
        raise ValueError('bound complete comparison source/data/kernel/start contract required')
    suffixes=('manifest.json','state.pt' if arm=='gru_sac' else 'model.zip')
    required=ROOT_FILES|{f'{name}/{suffix}' for name in ['initial']+[f'checkpoint-{s}' for s in range(10000,100001,10000)] for suffix in suffixes}
    entries=list(folder.rglob('*'))
    if (any(p.is_symlink() for p in entries) or {str(p.relative_to(folder)) for p in entries if p.is_file()}!=required
            or sum(p.stat().st_size for p in entries if p.is_file())>256*1024*1024):
        raise ValueError('complete bounded allowlisted cell artifacts required')
    history=[json.loads(line) for line in (folder/'learning-curve.jsonl').read_text().splitlines()]
    selected,final=read('selected.json'),read('final.json')
    best=verify_history(history,selected,final,arm)
    run=read('run.json')
    if (run['status']!='complete' or run['actual_transitions']!=100000
            or run['actual_updates']!=update_count(arm,100000)
            or run['selected_step']!=best['steps'] or run['selected_score']!=best['score']
            or run['final_score']!=history[-1]['score']):
        raise ValueError('complete actual run and selected/final identities required')
    if arm=='gru_sac':
        if run['config']!=asdict(candidate_config(seed)) or run['lineage']!=lineage or run['exposure_mode']!='unit_magnitude':
            raise ValueError('fixed GRU run configuration required')
    elif any(run.get(key)!=value for key,value in meta.items()) or run['actual_optimizer_steps']!=expected_counts(arm,100000)[1]:
        raise ValueError('fixed native run metadata and final actual optimizer counts required')
    with preserved_rng():
        for steps in [0]+list(range(10000,100001,10000)):
            name='initial' if steps==0 else f'checkpoint-{steps}'
            if arm=='gru_sac':
                model,_=load_checkpoint(folder/name,expected_lineage=lineage)
                if (model.updates!=update_count(arm,steps) or model.max_updates!=99000
                        or model.networks.observation_size!=21 or model.networks.hidden_size!=64
                        or model.learning_rate!=.0003 or model.gamma!=.99 or model.tau!=.005):
                    raise ValueError('actual bounded GRU checkpoint counters mismatch')
                if steps==0:
                    entry=initializations['entries'][f'{arm}-{seed}']
                    source,_=load_checkpoint(starts/entry['asset']['file'],expected_lineage=starts_lineage())
                    if (not torch.equal(model.log_alpha,source.log_alpha)
                            or any(not torch.equal(v,source.networks.state_dict()[k]) for k,v in model.networks.state_dict().items())):
                        raise ValueError('exact portable GRU initial tensors required')
            else:
                model=verify_native_checkpoint(folder/name,arm=arm,seed=seed,steps=steps,lineage=lineage)
                if steps==0:
                    entry=initializations['entries'][f'{arm}-{seed}']['asset']
                    arrays,_=read_initialization((starts/entry['file']).read_bytes(),entry['tensors'])
                    if any(not torch.equal(value,torch.from_numpy(arrays[key])) for key,value in model.policy.state_dict().items()):
                        raise ValueError('exact portable native initial tensors required')
        phases={}
        for phase,pointer in (('selected',selected),('final',final)):
            results=read(phase+'-validation.json')
            networks=None
            if arm=='gru_sac':
                restored,_=load_checkpoint(folder/pointer['checkpoint'],expected_lineage=lineage)
                networks=restored.networks
            verify_report(results,dataset,networks,observation_profile=spec['arm_contracts'][arm]['observation_profile'])
            if selection_score(results)!=pointer['score']:
                raise ValueError('saved selected/final reports disagree with curve/pointers')
            phases[phase]=phase_summary(results)
            if phase=='selected':
                selected_results=results
    for i,row in enumerate(history):
        if set(row['symbols'])!=set(selected_results):
            raise ValueError('exact development symbols required in every curve point')
        compact={key:dict(metrics=value,baselines=selected_results[key]['baselines'])
                 for key,value in row['symbols'].items()}
        expected=curve_record(compact,history[:i],row['steps'],'post_update',row['elapsed_seconds'],row['training'])
        expected.update(learner_updates=row['learner_updates'],checkpoint=row['checkpoint'])
        if canonical(expected)!=canonical(row):
            raise ValueError('curve totals/change/risk/cost metrics disagree with symbol records')
    with (folder/'learning-curve.csv').open(newline='') as stream:
        reader=csv.DictReader(stream)
        if set(reader.fieldnames or ())!={'steps','phase','return_pct','change_pp','equity_change_pct_from_previous',
                'gain_since_50k_pp','best_return_pct','buy_hold_return_pct','excess_return_pp',
                'mean_max_drawdown_pct','mean_costs','mean_turnover','elapsed_seconds'}:
            raise ValueError('complete compact curve CSV columns required')
        csv_rows=list(reader)
    if len(csv_rows)!=len(history) or any(
            any(value!=('' if history[i].get(key) is None else str(history[i][key])) for key,value in row.items())
            for i,row in enumerate(csv_rows)):
        raise ValueError('CSV must match compact checkpoint curve')
    for name in ('training-sampling.json','training-execution.json'):
        if read(name)['transitions']!=100000:
            raise ValueError('actual full training sampling/execution counts required')
    inventory={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in entries if p.is_file()}
    curve=[{key:row[key] for key in ('steps','return_pct','change_pp','excess_return_pp','mean_max_drawdown_pct','buy_hold_return_pct')} for row in history]
    return dict(arm=arm,seed=seed,saved_artifacts_verified=True,artifact_inventory_sha256=digest(inventory),curve=curve,selected=phases['selected'],final=phases['final'],
                source_hash=expected_source_hash,source_revision=expected_git_revision,
                prospective_confirmation_required=True,objective_achieved=False)

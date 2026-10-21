"""Contract fixtures only; no model fitting."""
import json,tarfile,hashlib
import pytest
from test_sac_state_artifacts import archive as state_archive
from test_diversity_jobs import counts
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.runtime.colab.jobs import recurrent_state_plan
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.training.diversity import protocol,sampling_summary
from rl_stocks.training.recurrent_state import overrides


def archive(root,arm='recurrent_full',smoke=True,mutate=None):
    state_archive(root);a=root/'artifacts';r=a/'runs/fixture'
    m=json.loads((r/'metadata.json').read_text());old=a/'frozen'/m.pop('bundle_id');f=json.loads((old/'manifest.json').read_text())
    p=recurrent_state_plan(arm,smoke=smoke);job=p['jobs'][0];c=counts('expanded',smoke)
    if smoke:
        c['transitions']=512;c['starts']=4
        c['history_groups']={k:v for k,v in list(c['history_groups'].items())[:4]}
    from rl_stocks.rl.registry.factory import resolved_parameters
    curve=[dict(steps=s,phase='post_update' if s==job['steps'] else 'on_step',score=.01) for s in range(job['evaluation_every'],job['steps']+1,job['evaluation_every'])]
    m.update({k:v for k,v in job.items() if k not in ('initialization','start_sampling')},dataset=p['dataset_contract']['identity'],
        requested_steps=job['steps'],actual_steps=job['steps'],resolved_parameters=resolved_parameters(job['algorithm'],smoke,overrides(arm,smoke)),
        start_sampling=sampling_summary('expanded',smoke),sampling_diagnostics=c,validation_checkpoints=curve,
        diversity_initialization=dict(alpha=1.,target_entropy=-4.,network_sha256={n:'a'*64 for n in ('actor','critic','critic_target')}))
    m.pop('diversity_initialization')
    from rl_stocks.training.recurrent_state import arm_spec
    params=overrides(arm,smoke); recurrent=job['algorithm']=='recurrent_ppo'
    record=dict(algorithm='RecurrentPPO' if recurrent else 'PPO',observation_size=21,action_size=4,
        num_timesteps=job['steps'],**{k:params[k] for k in ('n_steps','batch_size','n_epochs','gamma','gae_lambda','ent_coef','learning_rate')},
        net_arch=params['policy_kwargs']['net_arch'],actor_lstm=dict(hidden_size=32 if smoke else 64,num_layers=1) if recurrent else None,
        critic_lstm=recurrent,shared_lstm=False,policy_tensor_sha256='a'*64)
    for key in ('recurrent_state_initialization','recurrent_state_final','recurrent_state_selected'):m[key]=json.loads(json.dumps(record))
    m['recurrent_state_initialization']['num_timesteps']=0
    from rl_stocks.environment.observations.state import observation_spec
    m['observation']=observation_spec(arm_spec(arm)['observation_profile'])
    m['selected_step']=job['steps']
    f.update(metadata=m.copy(),normalizer=protocol()['frozen_normalizer'])
    if mutate:mutate(m,f)
    identity=digest(f);folder=old.with_name(identity);old.rename(folder);m['bundle_id']=identity
    (folder/'manifest.json').write_bytes(canonical(f));(folder/'checksums.json').unlink()
    (folder/'checksums.json').write_bytes(canonical({q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in folder.iterdir()}))
    (r/'metadata.json').write_text(json.dumps(m));(r/'training-sampling.json').write_text(json.dumps(c))
    (r/'learning-curve.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in curve))
    (r/'learning-curve.csv').write_text('steps,return_pct\n'+''.join(f"{v['steps']},1.0\n" for v in curve))
    (a/'completion.json').write_text(json.dumps(dict(plan=p,dataset=m['dataset'],python=[3,13,16],gpu=None,completed=[dict(bundle_id=identity)])))
    from rl_stocks.training.checkpoint_state import write_checkpoint_state
    report={'fixture':{'split':'validation','metrics':{'net_return':.01}}}
    (r/'best.zip').write_bytes(b'best')
    (r/'latest.zip').write_bytes(b'latest')
    (r/'selected-validation.json').write_text(json.dumps(report))
    write_checkpoint_state(r,report,'latest.zip',job['steps'],'post_update',job['steps'],'post_update',True)
    path=root/'recurrent.tar'
    with tarfile.open(path,'w') as t:t.add(a,arcname='artifacts')
    return path,p,m['dataset']


@pytest.mark.parametrize('arm',['feedforward_full','recurrent_masked','recurrent_full'])
@pytest.mark.parametrize('smoke',[True,False])
def test_registered_archives_accept(tmp_path,arm,smoke):
    assert len(validate_archive(*archive(tmp_path,arm,smoke)))==1


@pytest.mark.parametrize('mutate',[
    lambda m,f:m['recurrent_state_initialization'].update(observation_size=12),
    lambda m,f:m['recurrent_state_final'].update(num_timesteps=999),
    lambda m,f:m['recurrent_state_selected'].update(actor_lstm=None),
    lambda m,f:m['sampling_diagnostics'].update(transitions=999),
    lambda m,f:m.update(start_sampling={'mode':'legacy_uniform'}),
    lambda m,f:f['normalizer']['means'].__setitem__(0,999.),
    lambda m,f:m['resolved_parameters'].update(gamma=.95),
    lambda m,f:m['recurrent_state_final'].update(shared_lstm=True),
])
def test_semantic_corruption_rejected(tmp_path,mutate):
    with pytest.raises(RuntimeError):validate_archive(*archive(tmp_path,mutate=mutate))


def test_checkpoint_identity_required(tmp_path):
    _,plan,dataset=archive(tmp_path)
    (tmp_path/'artifacts/runs/fixture/checkpoint-state.json').unlink()
    path=tmp_path/'missing.tar'
    with tarfile.open(path,'w') as t:t.add(tmp_path/'artifacts',arcname='artifacts')
    with pytest.raises(RuntimeError,match='mandatory checkpoint identity'):validate_archive(path,plan,dataset)

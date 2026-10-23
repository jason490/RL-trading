"""Static retrieval contracts; no fixture claims real fitting or runtime measurements."""
import hashlib,json,shutil,tarfile
from copy import deepcopy
import pytest
from test_guard_artifacts import archive as guard_archive
from test_clipping_protocol import Dummy
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.rl.registry.factory import create,resolved_parameters
from rl_stocks.runtime.colab.jobs import clipping_plan,validate_plan
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.training.input_clipping import (protocol,overrides,model_record,apply_initialization,
    initialization_entry,ROOT,PROTOCOL_SHA256,runtime_projection)


def profile():
    durations=dict(control=10.,clip5=12.)
    return dict(version=1,protocol_sha256=PROTOCOL_SHA256,smoke_acceptance_file='.runtime/clipping-smoke-acceptance.json',
        smoke_acceptance_sha256='a'*64,durations_seconds=durations,projection=runtime_projection(durations))


def archive(root,arm='clip5',smoke=True,mutate=None):
    guard_archive(root,'control',smoke);a=root/'artifacts';run=a/'runs/fixture'
    m=json.loads((run/'metadata.json').read_text());old=a/'frozen'/m.pop('bundle_id');f=json.loads((old/'manifest.json').read_text())
    plan=clipping_plan(arm,smoke=smoke,profile=None if smoke else profile());job=plan['jobs'][0];params=overrides(arm,smoke)
    model=create('recurrent_ppo',Dummy(),smoke=smoke,overrides=params);proof=apply_initialization(model,arm,11,smoke)
    shutil.copyfile(ROOT/initialization_entry(arm,11,smoke)['file'],run/'clipping-initialization.npz')
    initial=model_record(model);final=deepcopy(initial);final.update(num_timesteps=job['steps'],policy_tensor_sha256='b'*64)
    nominal=job['steps']//params['n_steps']*params['n_epochs'];steps=nominal*params['n_steps']//params['batch_size']
    final['optimizer'].update(num_timesteps=job['steps'],sb3_nominal_epoch_counter=nominal,optimizer_state_count=21,
        adam_step_min=steps,adam_step_max=steps,per_parameter_adam_steps={n:steps for n,_ in model.policy.named_parameters()})
    for k in list(m):
        if k.startswith('guard_'):m.pop(k)
    m.update({k:v for k,v in job.items() if k not in ('initialization','start_sampling')},clipping_arm=arm,
        clipping_initialization=initial,clipping_final=deepcopy(final),clipping_selected=deepcopy(final),
        clipping_shared_initialization=proof,resolved_parameters=resolved_parameters('recurrent_ppo',smoke,params),duration_seconds=10.)
    c=m['sampling_diagnostics'];year='2010'
    per={key.split(':')[0]:dict(**row,decision_years={year:row['transitions']},execution_years={year:row['transitions']}) for key,row in c['history_groups'].items()}
    c['year_counts']=dict(version=1,measured=True,start_years={year:c['starts']},decision_years={year:job['steps']},execution_years={year:job['steps']},decision_execution_year_pairs={year+'->'+year:job['steps']},per_symbol=per)
    (run/'training-sampling.json').write_text(json.dumps(c))
    m['dataset']=protocol()['datasets'][arm]
    f['feature_schema']=protocol()['dataset_contracts'][arm]['schema']
    if arm=='clip5':f['normalizer']['clip_abs']=5.
    f['metadata']=deepcopy(m)
    if mutate:mutate(m,f)
    identity=digest(f);folder=old.with_name(identity);old.rename(folder);m['bundle_id']=identity
    (folder/'manifest.json').write_bytes(canonical(f));(folder/'checksums.json').unlink()
    (folder/'checksums.json').write_bytes(canonical({q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in folder.iterdir()}))
    (run/'metadata.json').write_text(json.dumps(m));(a/'completion.json').write_text(json.dumps(dict(plan=plan,dataset=m['dataset'],python=[3,13,16],gpu=None,completed=[dict(bundle_id=identity)])))
    packed=root/'clipping.tar'
    with tarfile.open(packed,'w') as t:t.add(a,arcname='artifacts')
    return packed,plan,m['dataset']


@pytest.mark.parametrize('arm',['control','clip5'])
@pytest.mark.parametrize('smoke',[True,False])
def test_both_full_width_archives_and_runtime_gated_plans_accept(tmp_path,arm,smoke):
    assert len(validate_archive(*archive(tmp_path,arm,smoke)))==1


@pytest.mark.parametrize('mutate',[
    lambda m,f:m.update(clipping_arm='control'),
    lambda m,f:m['clipping_final'].update(net_arch=[128,128]),
    lambda m,f:m['clipping_final'].update(actor_lstm=dict(hidden_size=128,num_layers=1)),
    lambda m,f:m['clipping_final'].update(target_kl=.01),
    lambda m,f:m['clipping_final']['optimizer'].update(parameter_count=221321),
    lambda m,f:m['clipping_final']['optimizer'].update(adam_step_max=999),
    lambda m,f:m['clipping_shared_initialization'].update(rng_sha256='0'*64),
    lambda m,f:m.update(duration_seconds=-1.),
    lambda m,f:m['clipping_final']['optimizer']['parameter_groups'][0].update(weight_decay=.0001),
    lambda m,f:m['clipping_final']['optimizer']['parameter_groups'][0].update(eps=1e-8),
    lambda m,f:m['clipping_final']['coupled_adam'].update(defaults_decoupled_weight_decay=True),
    lambda m,f:m['sampling_diagnostics'].pop('year_counts'),
    lambda m,f:m['action_execution'].update(evaluation_exposure_mode='unrestricted')])
def test_wrong_width_optimizer_or_initialization_rejects(tmp_path,mutate):
    with pytest.raises(RuntimeError):validate_archive(*archive(tmp_path,mutate=mutate))


@pytest.mark.parametrize('failure',['checkpoint','snapshot','snapshot_bytes'])
def test_mandatory_checkpoint_and_input_snapshot_evidence_rejects_corruption(tmp_path,failure):
    _,plan,dataset=archive(tmp_path);run=tmp_path/'artifacts/runs/fixture'
    if failure=='checkpoint':(run/'checkpoint-state.json').unlink()
    if failure=='snapshot':(run/'clipping-initialization.npz').unlink()
    if failure=='snapshot_bytes':
        p=run/'clipping-initialization.npz';p.write_bytes(p.read_bytes()[:-1])
    packed=tmp_path/'bad.tar'
    with tarfile.open(packed,'w') as t:t.add(tmp_path/'artifacts',arcname='artifacts')
    with pytest.raises(RuntimeError):validate_archive(packed,plan,dataset)


@pytest.mark.parametrize('arm',['control','clip5'])
def test_long_plan_requires_accepted_runtime_and_unchanged_floor(arm):
    with pytest.raises(ValueError):clipping_plan(arm)
    p=clipping_plan(arm,profile=profile());assert validate_plan(p)==p
    assert p['account_contract']==dict(minimum_balance=150,teardown_balance=170)
    for mutate in [lambda p:p['runtime_acceptance']['projection'].update(projected_seconds=0),
        lambda p:p['runtime_acceptance']['durations_seconds'].update(clip5=1000.),
        lambda p:p['jobs'][0]['overrides']['policy_kwargs'].update(lstm_hidden_size=256),
        lambda p:p['account_contract'].update(minimum_balance=0)]:
        bad=deepcopy(p);mutate(bad)
        with pytest.raises(ValueError):validate_plan(bad)


@pytest.mark.parametrize('mutate',[
    lambda m,f:f['normalizer'].update(clip_abs=4.),
    lambda m,f:f['normalizer'].pop('clip_abs'),
    lambda m,f:f.update(feature_schema='daily-indicators-v1'),
    lambda m,f:m.update(dataset=protocol()['datasets']['control']),
])
def test_candidate_cannot_drop_change_or_relabel_preprocessing(tmp_path,mutate):
    with pytest.raises(RuntimeError):validate_archive(*archive(tmp_path,mutate=mutate))

"""Static artifact contracts only; actual saved fitting remains a Colab gate."""
import json,tarfile,hashlib,shutil
from copy import deepcopy
import pytest
from test_alignment_artifacts import archive as alignment_archive
from test_hybrid_recurrent import Dummy
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.rl.registry.factory import create,resolved_parameters
from rl_stocks.rl.registry.distribution import distribution_spec
from rl_stocks.runtime.colab.jobs import guard_plan,validate_plan
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.training.recurrent_guard import protocol,overrides,model_record,modes,apply_shared_initialization,initialization_path


def archive(root,arm='kl_guard',smoke=True,mutate=None):
    alignment_archive(root,'posthoc_unit',smoke)
    a=root/'artifacts';r=a/'runs/fixture';m=json.loads((r/'metadata.json').read_text())
    old=a/'frozen'/m.pop('bundle_id');f=json.loads((old/'manifest.json').read_text())
    plan=guard_plan(arm,smoke=smoke);job=plan['jobs'][0];params=overrides(arm,smoke)
    model=create('recurrent_ppo',Dummy(),smoke=smoke,overrides=params)
    proof=apply_shared_initialization(model,11,smoke)
    shutil.copyfile(initialization_path(11,smoke),r/'guard-initialization.npz')
    initial=model_record(model);final=deepcopy(initial);final.update(num_timesteps=job['steps'],policy_tensor_sha256='b'*64)
    nominal=job['steps']//params['n_steps']*params['n_epochs'];upper=nominal*params['n_steps']//params['batch_size']
    steps=upper if arm=='control' else upper//2
    final['optimizer'].update(num_timesteps=job['steps'],sb3_nominal_epoch_counter=nominal,optimizer_state_count=21,
        adam_step_min=steps,adam_step_max=steps,per_parameter_adam_steps={n:steps for n,_ in model.policy.named_parameters()})
    for k in list(m):
        if k.startswith('alignment_'):m.pop(k)
    m.update({k:v for k,v in job.items() if k not in ('initialization','start_sampling')},
        resolved_parameters=resolved_parameters('recurrent_ppo',smoke,params),guard_initialization=initial,
        guard_final=deepcopy(final),guard_selected=deepcopy(final),guard_shared_initialization=proof,action_execution=modes(arm),action_distribution=distribution_spec())
    f['metadata']=m.copy()
    if mutate:mutate(m,f)
    identity=digest(f);folder=old.with_name(identity);old.rename(folder);m['bundle_id']=identity
    (folder/'manifest.json').write_bytes(canonical(f));(folder/'checksums.json').unlink()
    (folder/'checksums.json').write_bytes(canonical({q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in folder.iterdir()}))
    (r/'metadata.json').write_text(json.dumps(m))
    (a/'completion.json').write_text(json.dumps(dict(plan=plan,dataset=m['dataset'],python=[3,13,16],gpu=None,completed=[dict(bundle_id=identity)])))
    path=root/'guard.tar'
    with tarfile.open(path,'w') as t:t.add(a,arcname='artifacts')
    return path,plan,m['dataset']


@pytest.mark.parametrize('arm',protocol()['arms'])
@pytest.mark.parametrize('smoke',[True,False])
def test_registered_guard_archives_accept(tmp_path,arm,smoke):
    assert len(validate_archive(*archive(tmp_path,arm,smoke)))==1


@pytest.mark.parametrize('mutate',[
    lambda m,f:m['guard_final'].update(target_kl=None),
    lambda m,f:m['guard_initialization'].update(clip_range=.1),
    lambda m,f:m['guard_final']['optimizer'].update(adam_step_max=999999),
    lambda m,f:m['guard_final']['optimizer'].update(optimizer_state_count=0),
    lambda m,f:m['guard_selected']['optimizer'].update(all_parameters_and_moments_finite=False),
    lambda m,f:m['guard_final'].update(policy_tensor_sha256=m['guard_initialization']['policy_tensor_sha256']),
    lambda m,f:m['action_distribution'].update(version=True),
    lambda m,f:m['guard_shared_initialization'].update(rng_sha256='0'*64),
    lambda m,f:m['guard_initialization'].update(policy_tensor_sha256='a'*64),
    lambda m,f:m['action_execution'].update(evaluation_exposure_mode='unrestricted'),
])
def test_guard_semantic_corruption_rejected(tmp_path,mutate):
    with pytest.raises(RuntimeError):validate_archive(*archive(tmp_path,mutate=mutate))


@pytest.mark.parametrize('arm',protocol()['arms'])
@pytest.mark.parametrize('smoke',[True,False])
def test_guard_jobs_bind_only_fixed_target_and_account_floor(arm,smoke):
    p=guard_plan(arm,smoke=smoke);assert validate_plan(p)==p
    assert p['account_contract']=={'minimum_balance':150,'teardown_balance':170}
    for change in [lambda p:p['jobs'][0]['overrides'].update(target_kl=.02),
                   lambda p:p['jobs'][0].update(evaluation_exposure_mode='unrestricted'),
                   lambda p:p['jobs'][0].update(steps=250000),
                   lambda p:p['account_contract'].update(minimum_balance=0)]:
        bad=deepcopy(p);change(bad)
        with pytest.raises(ValueError):validate_plan(bad)


def test_guard_requires_checkpoint_identity_even_if_other_artifacts_match(tmp_path):
    _,plan,dataset=archive(tmp_path)
    (tmp_path/'artifacts/runs/fixture/checkpoint-state.json').unlink()
    path=tmp_path/'missing.tar'
    with tarfile.open(path,'w') as t:t.add(tmp_path/'artifacts',arcname='artifacts')
    with pytest.raises(RuntimeError,match='mandatory checkpoint identity'):validate_archive(path,plan,dataset)


def test_shared_initialization_snapshot_is_required_and_checksummed(tmp_path):
    _,plan,dataset=archive(tmp_path)
    path=tmp_path/'artifacts/runs/fixture/guard-initialization.npz'
    path.write_bytes(path.read_bytes()[:-1])
    packed=tmp_path/'bad-initialization.tar'
    with tarfile.open(packed,'w') as t:t.add(tmp_path/'artifacts',arcname='artifacts')
    with pytest.raises(RuntimeError):validate_archive(packed,plan,dataset)

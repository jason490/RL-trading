"""Static retrieval contracts; actual fitting acceptance remains Colab-only."""
import hashlib,json,shutil,tarfile
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from gymnasium import Env,spaces
from test_guard_artifacts import archive as guard_archive
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.rl.registry.factory import create,resolved_parameters
from rl_stocks.runtime.colab.jobs import session_plan,validate_plan
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.training.session_context import protocol,overrides,arm_spec,model_record,apply_initialization,initialization_entry,ROOT


def archive(root,arm='full',smoke=True,mutate=None):
    guard_archive(root,'control',smoke);a=root/'artifacts';run=a/'runs/fixture'
    m=json.loads((run/'metadata.json').read_text());old=a/'frozen'/m.pop('bundle_id');f=json.loads((old/'manifest.json').read_text())
    plan=session_plan(arm,smoke=smoke);job=plan['jobs'][0];params=overrides(arm,smoke)
    class Dummy(Env):
        observation_space=spaces.Box(-np.inf,np.inf,(arm_spec(arm)['observation_size'],),dtype=np.float32)
        action_space=spaces.Box(-1,1,(4,),dtype=np.float32)
    model=create('recurrent_ppo',Dummy(),smoke=smoke,overrides=params);proof=apply_initialization(model,arm,11,smoke)
    shutil.copyfile(ROOT/initialization_entry(arm,11,smoke)['file'],run/'session-initialization.npz')
    initial=model_record(model);final=deepcopy(initial);final.update(num_timesteps=job['steps'],policy_tensor_sha256='b'*64)
    nominal=job['steps']//params['n_steps']*params['n_epochs'];steps=nominal*params['n_steps']//params['batch_size']
    final['optimizer'].update(num_timesteps=job['steps'],sb3_nominal_epoch_counter=nominal,optimizer_state_count=21,
        adam_step_min=steps,adam_step_max=steps,per_parameter_adam_steps={n:steps for n,_ in model.policy.named_parameters()})
    for k in list(m):
        if k.startswith('guard_'):m.pop(k)
    m.update({k:v for k,v in job.items() if k not in ('initialization','start_sampling')},dataset=plan['dataset_contract']['identity'],
        session_context_arm=arm,session_context_initialization=initial,session_context_final=deepcopy(final),session_context_selected=deepcopy(final),
        session_context_shared_initialization=proof,resolved_parameters=resolved_parameters('recurrent_ppo',smoke,params,'native',arm_spec(arm)['market_features']))
    contract=protocol()['dataset_contracts'][arm]
    normalizers=json.loads((Path(__file__).parent/'fixtures/session_normalizers.json').read_text())
    assert digest(normalizers[arm])==contract['normalizer_sha256']
    f.update(metadata=m.copy(),names=contract['names'],feature_schema=contract['schema'],normalizer=normalizers[arm])
    if mutate:mutate(m,f)
    identity=digest(f);folder=old.with_name(identity);old.rename(folder);m['bundle_id']=identity
    (folder/'manifest.json').write_bytes(canonical(f));(folder/'checksums.json').unlink()
    (folder/'checksums.json').write_bytes(canonical({q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in folder.iterdir()}))
    (run/'metadata.json').write_text(json.dumps(m));(a/'completion.json').write_text(json.dumps(dict(plan=plan,dataset=m['dataset'],python=[3,13,16],gpu=None,completed=[dict(bundle_id=identity)])))
    packed=root/'session.tar'
    with tarfile.open(packed,'w') as t:t.add(a,arcname='artifacts')
    return packed,plan,m['dataset']


@pytest.mark.parametrize('arm',['legacy','masked','full'])
@pytest.mark.parametrize('smoke',[True,False])
def test_three_registered_dataset_dimension_archives_accept(tmp_path,arm,smoke):
    assert len(validate_archive(*archive(tmp_path,arm,smoke)))==1


@pytest.mark.parametrize('mutate',[
    lambda m,f:m.update(session_context_arm='masked'),
    lambda m,f:m['session_context_final'].update(observation_size=21),
    lambda m,f:m['session_context_final'].update(target_kl=.01),
    lambda m,f:m['session_context_final']['optimizer'].update(adam_step_max=999),
    lambda m,f:f.update(feature_schema='daily-indicators-v1'),
    lambda m,f:f['names'].reverse(),
    lambda m,f:f['normalizer']['means'].__setitem__(0,100.),
    lambda m,f:m['session_context_shared_initialization'].update(rng_sha256='0'*64),
    lambda m,f:m['action_execution'].update(evaluation_exposure_mode='unrestricted')])
def test_semantic_contract_corruption_rejects(tmp_path,mutate):
    with pytest.raises(RuntimeError):validate_archive(*archive(tmp_path,mutate=mutate))


@pytest.mark.parametrize('arm',['legacy','masked','full'])
def test_plans_pin_dataset_initialization_and_account_floor(arm):
    p=session_plan(arm,smoke=True);assert validate_plan(p)==p
    assert p['account_contract']==dict(minimum_balance=150,teardown_balance=170)
    for mutate in [lambda p:p['dataset_contract'].update(identity='0'*64),lambda p:p['initialization_contract'].update(rng_sha256='0'*64),
        lambda p:p['jobs'][0]['overrides'].update(target_kl=.01),lambda p:p['jobs'][0].update(steps=250000)]:
        bad=deepcopy(p);mutate(bad)
        with pytest.raises(ValueError):validate_plan(bad)


@pytest.mark.parametrize('failure',['checkpoint','snapshot','snapshot_bytes'])
def test_mandatory_evidence_is_retained_and_checksummed(tmp_path,failure):
    _,plan,dataset=archive(tmp_path)
    run=tmp_path/'artifacts/runs/fixture'
    if failure=='checkpoint':(run/'checkpoint-state.json').unlink()
    if failure=='snapshot':(run/'session-initialization.npz').unlink()
    if failure=='snapshot_bytes':
        path=run/'session-initialization.npz';path.write_bytes(path.read_bytes()[:-1])
    packed=tmp_path/'missing.tar'
    with tarfile.open(packed,'w') as t:t.add(tmp_path/'artifacts',arcname='artifacts')
    with pytest.raises(RuntimeError):validate_archive(packed,plan,dataset)

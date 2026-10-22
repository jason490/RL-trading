"""Synthetic archive contracts; real model restoration is a separate smoke gate."""
import json,tarfile,hashlib
from copy import deepcopy
import pytest
from test_alignment_artifacts import archive as alignment_archive
from test_hybrid_recurrent import Dummy
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.rl.registry.factory import create,resolved_parameters
from rl_stocks.runtime.colab.jobs import cash_plan
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.training.recurrent_cash import protocol,arm_spec,overrides,model_record,modes


def archive(root,arm='categorical_cash',smoke=True,mutate=None):
    alignment_archive(root,'aligned_unit' if arm=='gaussian_unit' else 'posthoc_unit',smoke)
    a=root/'artifacts';r=a/'runs/fixture';m=json.loads((r/'metadata.json').read_text())
    old=a/'frozen'/m.pop('bundle_id');f=json.loads((old/'manifest.json').read_text())
    plan=cash_plan(arm,smoke=smoke);job=plan['jobs'][0];params=overrides(arm,smoke)
    model=create(job['algorithm'],Dummy(),smoke=smoke,overrides=params)
    initial=model_record(model);final=deepcopy(initial)
    final.update(num_timesteps=job['steps'],policy_tensor_sha256='b'*64,
        optimizer_updates=job['steps']//params['n_steps']*params['n_epochs'],optimizer_parameter_states=1)
    for k in list(m):
        if k.startswith('alignment_'):m.pop(k)
    m.update({k:v for k,v in job.items() if k not in ('initialization','start_sampling')},
        resolved_parameters=resolved_parameters(job['algorithm'],smoke,params),
        cash_initialization=initial,cash_final=deepcopy(final),cash_selected=deepcopy(final),
        action_execution=modes(arm),action_distribution=initial['action_distribution'])
    categorical=arm.startswith('categorical')
    if categorical:m['training_execution'].update(mode='unrestricted',requested_nonzero=340 if arm=='categorical_cash' else job['steps'],processed_nonunit=0,sizing_changed=0)
    trace={'raw_action':[-1.,1.,1.,1.]}
    if not categorical:trace.update(policy_action=[-.25,1.,1.,1.],exposure_mode='unit_magnitude')
    report={'fixture':{'split':'validation','metrics':{'net_return':.01},'rows':[trace]}}
    f['metadata']=m.copy()
    if mutate:mutate(m,f)
    identity=digest(f);folder=old.with_name(identity);old.rename(folder);m['bundle_id']=identity
    (folder/'manifest.json').write_bytes(canonical(f));(folder/'checksums.json').unlink()
    (folder/'checksums.json').write_bytes(canonical({q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in folder.iterdir()}))
    (r/'metadata.json').write_text(json.dumps(m));(r/'training-execution.json').write_text(json.dumps(m['training_execution']))
    for name in ('selected-validation.json','validation.json'):(r/name).write_text(json.dumps(report))
    from rl_stocks.training.checkpoint_state import write_checkpoint_state
    write_checkpoint_state(r,report,'latest.zip',job['steps'],'post_update',job['steps'],'post_update',True)
    (a/'completion.json').write_text(json.dumps(dict(plan=plan,dataset=m['dataset'],python=[3,13,16],gpu=None,completed=[dict(bundle_id=identity)])))
    path=root/'cash.tar'
    with tarfile.open(path,'w') as t:t.add(a,arcname='artifacts')
    return path,plan,m['dataset']


@pytest.mark.parametrize('arm',protocol()['arms'])
@pytest.mark.parametrize('smoke',[True,False])
def test_cash_archives_accept(tmp_path,arm,smoke):
    assert len(validate_archive(*archive(tmp_path,arm,smoke)))==1


@pytest.mark.parametrize('mutate',[
    lambda m,f:m['cash_initialization'].update(parameter_count=1),
    lambda m,f:m['cash_final'].update(optimizer_updates=0),
    lambda m,f:m['cash_final'].update(optimizer_parameter_states=0),
    lambda m,f:m['cash_final'].update(policy_tensor_sha256=m['cash_initialization']['policy_tensor_sha256']),
    lambda m,f:m['cash_selected'].update(num_timesteps=999),
    lambda m,f:m['training_execution'].update(requested_nonzero=m['actual_steps']),
    lambda m,f:m['action_distribution'].update(cash_enabled=False),
    lambda m,f:m['cash_final']['action_distribution'].update(directions=[-1,False,True]),
    lambda m,f:m.update(action_execution={'version':1,'training_exposure_mode':'unit_magnitude','evaluation_exposure_mode':'unit_magnitude'}),
])
def test_cash_semantic_corruption_rejects(tmp_path,mutate):
    with pytest.raises(RuntimeError):validate_archive(*archive(tmp_path,mutate=mutate))

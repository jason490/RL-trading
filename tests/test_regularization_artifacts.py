"""Contract fixtures only; no model fitting."""
import json,tarfile,hashlib
import pytest
from test_sac_state_artifacts import archive as state_archive
from test_diversity_jobs import counts
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.runtime.colab.jobs import regularization_plan
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.training.diversity import protocol,sampling_summary
from rl_stocks.training.sac_regularization import overrides


def archive(root,arm='control',smoke=True,mutate=None):
    state_archive(root);a=root/'artifacts';r=a/'runs/fixture'
    m=json.loads((r/'metadata.json').read_text());old=a/'frozen'/m.pop('bundle_id');f=json.loads((old/'manifest.json').read_text())
    p=regularization_plan(arm,smoke=smoke);job=p['jobs'][0];c=counts('expanded',smoke)
    from rl_stocks.rl.registry.factory import resolved_parameters
    curve=[dict(steps=s,phase='post_update' if s==job['steps'] else 'on_step',score=.01) for s in range(job['evaluation_every'],job['steps']+1,job['evaluation_every'])]
    m.update({k:v for k,v in job.items() if k not in ('initialization','start_sampling')},dataset=p['dataset_contract']['identity'],
        requested_steps=job['steps'],actual_steps=job['steps'],resolved_parameters=resolved_parameters('sac',smoke,overrides(arm,smoke)),
        start_sampling=sampling_summary('expanded',smoke),sampling_diagnostics=c,validation_checkpoints=curve,
        diversity_initialization=dict(alpha=1.,target_entropy=-4.,network_sha256={n:'a'*64 for n in ('actor','critic','critic_target')}))
    initial=m.pop('diversity_initialization')
    from rl_stocks.training.sac_regularization import protocol as reg_protocol
    record=dict(initialization=initial,optimizers={n:dict(**{'class':'torch.optim.adam.Adam'},weight_decay=[0. if n=='entropy' else reg_protocol()['weight_decay'][arm]]) for n in ('actor','critic','entropy')})
    for key in ('regularization_initialization','regularization_final','regularization_selected'):m[key]=json.loads(json.dumps(record))
    f.update(metadata=m.copy(),normalizer=protocol()['frozen_normalizer'])
    if mutate:mutate(m,f)
    identity=digest(f);folder=old.with_name(identity);old.rename(folder);m['bundle_id']=identity
    (folder/'manifest.json').write_bytes(canonical(f));(folder/'checksums.json').unlink()
    (folder/'checksums.json').write_bytes(canonical({q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in folder.iterdir()}))
    (r/'metadata.json').write_text(json.dumps(m));(r/'training-sampling.json').write_text(json.dumps(c))
    (r/'learning-curve.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in curve))
    (r/'learning-curve.csv').write_text('steps,return_pct\n'+''.join(f"{v['steps']},1.0\n" for v in curve))
    (a/'completion.json').write_text(json.dumps(dict(plan=p,dataset=m['dataset'],python=[3,13,16],gpu=None,completed=[dict(bundle_id=identity)])))
    path=root/'diversity.tar'
    with tarfile.open(path,'w') as t:t.add(a,arcname='artifacts')
    return path,p,m['dataset']


@pytest.mark.parametrize('arm',['control','weight_decay'])
@pytest.mark.parametrize('smoke',[True,False])
def test_smoke_and_screen_contracts_accept(tmp_path,arm,smoke):
    assert len(validate_archive(*archive(tmp_path,arm,smoke)))==1


@pytest.mark.parametrize('mutate',[
    lambda m,f:m['regularization_initialization']['initialization'].update(alpha=.001),
    lambda m,f:m['regularization_final']['optimizers']['entropy'].update(weight_decay=[.001]),
    lambda m,f:m['regularization_selected']['optimizers']['actor'].update(weight_decay=[.1]),
    lambda m,f:m['sampling_diagnostics'].update(transitions=999),
    lambda m,f:m.update(start_sampling={'mode':'legacy_uniform'}),
    lambda m,f:f['normalizer']['means'].__setitem__(0,999.),
    lambda m,f:m['resolved_parameters'].update(ent_coef='auto_0.001'),
])
def test_semantic_changes_fail_despite_recomputed_checksums(tmp_path,mutate):
    with pytest.raises(RuntimeError):validate_archive(*archive(tmp_path,mutate=mutate))

"""Metadata/checksum fixtures only; no local training."""
import json, tarfile, hashlib
import pytest
from test_sac_state_artifacts import archive as state_archive
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.runtime.colab.jobs import td3_state_plan
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.rl.registry.factory import resolved_parameters
from rl_stocks.training.initialization import initialization_spec
from rl_stocks.training.td3_state import protocol


def archive(root,arm='execution_full',smoke=True,mutate=None):
    state_archive(root,arm); a=root/'artifacts'; r=a/'runs/fixture'
    m=json.loads((r/'metadata.json').read_text()); old=a/'frozen'/m.pop('bundle_id')
    f=json.loads((old/'manifest.json').read_text()); p=td3_state_plan(arm,smoke=smoke); job=p['jobs'][0]
    curve=[dict(steps=s,phase='post_update' if s==job['steps'] else 'on_step',score=.01) for s in range(job['evaluation_every'],job['steps']+1,job['evaluation_every'])]
    m.update({k:v for k,v in job.items() if k not in ('initialization','start_sampling')},
             requested_steps=job['steps'],actual_steps=job['steps'],
             initialization=initialization_spec('standard','td3'),
             resolved_parameters=resolved_parameters('td3',smoke),validation_checkpoints=curve,
             td3_initialization=dict(network_sha256={n:'a'*64 for n in ('actor','actor_target','critic','critic_target')},
                 exploration_noise=protocol()['exploration_noise'],policy_delay=2,target_policy_noise=.2,target_noise_clip=.5))
    f['metadata']=m.copy()
    if mutate:mutate(m,f)
    identity=digest(f); folder=old.with_name(identity); old.rename(folder); m['bundle_id']=identity
    (folder/'manifest.json').write_bytes(canonical(f)); (folder/'checksums.json').unlink()
    (folder/'checksums.json').write_bytes(canonical({q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in folder.iterdir()}))
    (r/'metadata.json').write_text(json.dumps(m))
    (r/'learning-curve.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in curve))
    (r/'learning-curve.csv').write_text('steps,return_pct\n'+''.join(f"{v['steps']},1.0\n" for v in curve))
    (a/'completion.json').write_text(json.dumps(dict(plan=p,dataset=m['dataset'],python=[3,13,16],gpu=None,completed=[dict(bundle_id=identity)])))
    path=root/'td3.tar'
    with tarfile.open(path,'w') as t:t.add(a,arcname='artifacts')
    return path,p,m['dataset']


@pytest.mark.parametrize('arm',['legacy','execution_masked','execution_full'])
@pytest.mark.parametrize('smoke',[False,True])
def test_both_smoke_and_full_td3_contracts_accepted(tmp_path,arm,smoke):
    assert len(validate_archive(*archive(tmp_path,arm,smoke)))==1


@pytest.mark.parametrize('change',[
    lambda m,f:m['td3_initialization']['exploration_noise'].update(sigma=[0.,0.,0.,0.]),
    lambda m,f:m['td3_initialization'].update(policy_delay=1),
    lambda m,f:m['td3_initialization']['network_sha256'].pop('actor_target'),
    lambda m,f:m['resolved_parameters'].update(batch_size=256),
    lambda m,f:m.update(study_protocol_sha256='wrong'),
    lambda m,f:f['metadata'].pop('td3_initialization'),
])
def test_semantic_changes_fail_despite_recomputed_checksums(tmp_path,change):
    with pytest.raises(RuntimeError):validate_archive(*archive(tmp_path,mutate=change))

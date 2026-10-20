"""Numerical manifest fixtures only; these tests never train a policy."""
import hashlib,json,tarfile
from pathlib import Path
import pytest
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.runtime.colab.jobs import history_plan
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.training.history.spec import PROTOCOL_SHA256,validate_counts
from rl_stocks.training.initialization import initialization_spec
from rl_stocks.environment.observations.state import observation_spec
from rl_stocks.rl.ppo.architecture import architecture_spec
from rl_stocks.rl.registry.factory import resolved_parameters
from rl_stocks.features.registry.assemble import NAMES
from test_history_jobs import counts


def fixture_archive(root,change=None):
    plan=history_plan('original',smoke=True);job=plan['jobs'][0];data=plan['dataset_contract']['identity'];c=counts()
    curve=[dict(steps=256,phase='on_step',score=.01),dict(steps=512,phase='post_update',score=.02)]
    run=dict(job,requested_steps=512,actual_steps=512,status='complete',dataset=data,
        initialization=initialization_spec('wide','ppo'),observation=observation_spec('execution_full'),
        start_sampling=validate_counts(c,'original',True,512),sampling_diagnostics=c,
        policy_architecture=architecture_spec('native',7),resolved_parameters=resolved_parameters('ppo',True,None,'native',7),
        reward_config=dict(kind='net_log_equity',scale=1.),validation_checkpoints=curve,
        history_augmentation=dict(arm='original'),history_protocol_sha256=PROTOCOL_SHA256)
    frozen=dict(names=list(NAMES),feature_schema='daily-indicators-v1',data_manifest=dict(history_augmentation=dict(arm='original')),metadata=run.copy())
    if change:change(run,frozen,c)
    identity=digest(frozen);run['bundle_id']=identity
    a=root/'artifacts';f=a/'frozen'/identity;f.mkdir(parents=True);r=a/'runs/fixture';r.mkdir(parents=True)
    (f/'manifest.json').write_bytes(canonical(frozen));(f/'model.zip').write_bytes(b'fixture only')
    (f/'checksums.json').write_bytes(canonical({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in f.iterdir()}))
    (r/'metadata.json').write_text(json.dumps(run));(r/'training-sampling.json').write_text(json.dumps(c))
    (r/'learning-curve.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in curve));(r/'learning-curve.csv').write_text('steps,return_pct\n256,1.0\n512,2.0\n');(r/'learning-curve.html').write_text('fixture')
    (a/'completion.json').write_text(json.dumps(dict(plan=plan,dataset=data,python=[3,13,16],gpu=None,completed=[dict(bundle_id=identity)])))
    path=root/'result.tar.gz'
    with tarfile.open(path,'w:gz') as t:t.add(a,arcname='artifacts')
    return path,plan,data


def test_history_archive_accepts_complete_contract(tmp_path):
    assert len(validate_archive(*fixture_archive(tmp_path)))==1


@pytest.mark.parametrize('change',[
    lambda r,f,c:c['history_groups']['SPY:recent'].update(starts=6),
    lambda r,f,c:r.update(history_protocol_sha256='wrong'),
    lambda r,f,c:f['data_manifest']['history_augmentation'].update(arm='older_etf'),
    lambda r,f,c:f.update(names=list(reversed(NAMES))),
])
def test_valid_checksums_do_not_hide_contract_tampering(tmp_path,change):
    with pytest.raises(RuntimeError):validate_archive(*fixture_archive(tmp_path,change))

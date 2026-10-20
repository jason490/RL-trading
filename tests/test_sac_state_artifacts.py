"""Checksummed manifest fixtures only; no local model fitting."""
import json,tarfile,hashlib
import pytest
from test_history_artifacts import fixture_archive
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.runtime.colab.jobs import sac_state_plan
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.rl.registry.factory import resolved_parameters
from rl_stocks.training.initialization import initialization_spec
from rl_stocks.environment.observations.state import observation_spec


def archive(root,arm='execution_full',mutate=None):
    fixture_archive(root);a=root/'artifacts';rp=a/'runs/fixture';m=json.loads((rp/'metadata.json').read_text())
    old=a/'frozen'/m['bundle_id'];f=json.loads((old/'manifest.json').read_text())
    p=sac_state_plan(arm,smoke=True);job=p['jobs'][0]
    for k in ['bundle_id','history_augmentation','history_protocol_sha256']:m.pop(k)
    curve=[dict(steps=512,phase='on_step',score=.01),dict(steps=1024,phase='post_update',score=.02)]
    m.update(job,requested_steps=1024,actual_steps=1024,dataset=p['dataset_contract']['identity'],initialization=initialization_spec('standard','sac'),observation=observation_spec(arm),start_sampling={'mode':'legacy_uniform'},sampling_diagnostics=None,resolved_parameters=resolved_parameters('sac',True,None,'native',7),validation_checkpoints=curve)
    f.update(metadata=m.copy(),data_manifest=dict(included_splits=['train','validation']))
    if mutate:mutate(m,f)
    identity=digest(f);m['bundle_id']=identity;folder=old.with_name(identity);old.rename(folder)
    (folder/'manifest.json').write_bytes(canonical(f));(folder/'checksums.json').unlink()
    (folder/'checksums.json').write_bytes(canonical({q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in folder.iterdir()}))
    (rp/'metadata.json').write_text(json.dumps(m));(rp/'learning-curve.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in curve));(rp/'learning-curve.csv').write_text('steps,return_pct\n512,1.0\n1024,2.0\n')
    (a/'completion.json').write_text(json.dumps(dict(plan=p,dataset=m['dataset'],python=[3,13,16],gpu=None,completed=[dict(bundle_id=identity)])))
    path=root/'sac.tar'
    with tarfile.open(path,'w') as t:t.add(a,arcname='artifacts')
    return path,p,m['dataset']


@pytest.mark.parametrize('arm',['legacy','execution_masked','execution_full'])
def test_all_fixed_observation_contracts_accepted(tmp_path,arm):
    assert len(validate_archive(*archive(tmp_path,arm)))==1


@pytest.mark.parametrize('mutate',[lambda m,f:m.update(observation=observation_spec('legacy')),lambda m,f:m.update(study_protocol_sha256='wrong'),lambda m,f:m['resolved_parameters'].update(batch_size=256),lambda m,f:f.update(names=['wrong'])])
def test_semantic_tampering_fails_despite_recomputed_checksums(tmp_path,mutate):
    with pytest.raises(RuntimeError):validate_archive(*archive(tmp_path,mutate=mutate))

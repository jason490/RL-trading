"""Manifest fixtures test contracts without training policy weights."""
import hashlib,json,tarfile
import pytest
from test_sac_state_artifacts import archive as state_archive
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.runtime.colab.jobs import temperature_plan
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.rl.registry.factory import resolved_parameters
from rl_stocks.training.sac_temperature import protocol
from rl_stocks.training.initialization import initialization_spec


def fixture(root,arm,mutate=None):
    state_archive(root);a=root/'artifacts';r=a/'runs/fixture';m=json.loads((r/'metadata.json').read_text());old=a/'frozen'/m.pop('bundle_id');f=json.loads((old/'manifest.json').read_text())
    p=temperature_plan(arm,smoke=True);m.update({k:v for k,v in p['jobs'][0].items() if k not in ('initialization','start_sampling','overrides')},initialization=initialization_spec('standard','sac'),resolved_parameters=resolved_parameters('sac',True,p['jobs'][0]['overrides'],'native',7),temperature_initialization=dict(alpha=protocol()['initial_alpha'][arm],target_entropy=-4.,network_sha256={k:'a'*64 for k in ('actor','critic','critic_target')}))
    f['metadata']=m.copy()
    if mutate:mutate(m,f)
    identity=digest(f);folder=old.with_name(identity);old.rename(folder);m['bundle_id']=identity
    (folder/'manifest.json').write_bytes(canonical(f));(folder/'checksums.json').unlink();(folder/'checksums.json').write_bytes(canonical({q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in folder.iterdir()}))
    (r/'metadata.json').write_text(json.dumps(m));(a/'completion.json').write_text(json.dumps(dict(plan=p,dataset=m['dataset'],python=[3,13,16],gpu=None,completed=[dict(bundle_id=identity)])))
    path=root/'temp.tar'
    with tarfile.open(path,'w') as t:t.add(a,arcname='artifacts')
    return path,p,m['dataset']


@pytest.mark.parametrize('arm',['default','cold_start'])
def test_temperature_archive_accepts_exact_arms(tmp_path,arm):
    assert len(validate_archive(*fixture(tmp_path,arm)))==1


@pytest.mark.parametrize('mutate',[lambda m,f:m['temperature_initialization'].update(alpha=1.),lambda m,f:m['temperature_initialization'].update(target_entropy=-2.),lambda m,f:m['resolved_parameters'].update(ent_coef='auto'),lambda m,f:m.update(study_protocol_sha256='wrong')])
def test_changed_temperature_semantics_rejected(tmp_path,mutate):
    with pytest.raises(RuntimeError):validate_archive(*fixture(tmp_path,'cold_start',mutate))

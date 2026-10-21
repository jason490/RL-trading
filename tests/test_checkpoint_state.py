import json
from pathlib import Path
from types import SimpleNamespace
import tarfile
import pytest
from rl_stocks.training.runner import ValidationCheckpoint
from rl_stocks.training.checkpoint_state import read_checkpoint_state


def result(score):
    return {'A':dict(split='validation',symbol='A',metrics=dict(net_return=score,max_drawdown=-.1,sharpe=0.,costs=1.,turnover=2.),baselines={'buy_hold':{'metrics':{'net_return':.3}}})}


@pytest.mark.parametrize('exposure_mode',['unrestricted','unit_magnitude'])
def test_worse_checkpoint_exports_its_own_paths_without_replacing_best(tmp_path,monkeypatch,exposure_mode):
    import rl_stocks.training.runner as runner
    run=tmp_path/'artifacts/runs/fixture';run.mkdir(parents=True)
    archive=tmp_path/'snapshot.tar.gz';monkeypatch.setenv('RL_CHECKPOINT_ARCHIVE',str(archive))
    scores=iter([.2,.05,.04])
    def evaluate(*args,**kwargs):
        assert args[-1]==exposure_mode
        return result(next(scores))
    monkeypatch.setattr(runner,'evaluate',evaluate)
    monkeypatch.setattr(runner,'render_curve',lambda *a:None)
    def report(results,destination,*a):
        destination.write_text('html')
        destination.with_suffix('.json').write_text(json.dumps(results))
    monkeypatch.setattr(runner,'report',report)
    cb=ValidationCheckpoint({},None,run,10,compact=True,target_steps=20,exposure_mode=exposure_mode)
    cb.model=SimpleNamespace(logger=SimpleNamespace(name_to_value={}),save=lambda p:Path(str(p)+'.zip').write_text(str(cb.num_timesteps)))
    cb.num_timesteps=10;cb.validate()
    cb.num_timesteps=20;cb.validate()
    state=read_checkpoint_state(run)
    assert state['latest']['steps']==20 and state['selected']['steps']==10
    assert state['score']==.05 and not state['final_post_update']
    assert json.loads((run/'selected-validation.json').read_text())['A']['metrics']['net_return']==.2
    assert json.loads((run/'checkpoint-validation.json').read_text())['A']['metrics']['net_return']==.05
    # Reproduce the old failure: legacy report stays at best while the latest
    # weights move on. The explicit manifest binds the new report correctly.
    assert json.loads((run/'validation.json').read_text())['A']['metrics']['net_return']==.2
    with tarfile.open(archive) as t:
        t.extractall(tmp_path/'retrieved',filter='data')
    assert read_checkpoint_state(tmp_path/'retrieved/artifacts/runs/fixture')==state
    cb.validate(phase='post_update')
    assert read_checkpoint_state(run)['final_post_update']
    assert read_checkpoint_state(run)['score']==.04
    (run/'checkpoint-validation.json').write_text(json.dumps(result(.9)))
    with pytest.raises(ValueError,match='checksum'):read_checkpoint_state(run)


def test_unsafe_manifest_path_is_rejected(tmp_path):
    from rl_stocks.training.checkpoint_state import validate_checkpoint_state
    state={'version':1,'final_post_update':False,'latest':{'steps':1,'phase':'on_step','model':'../weights'},'selected':{}}
    with pytest.raises(ValueError,match='basename'):validate_checkpoint_state(tmp_path,state)


@pytest.mark.parametrize('tamper',[None,'report','final_flag'])
def test_completed_archive_checks_checkpoint_identity_when_present(tmp_path,tamper):
    from test_regularization_artifacts import archive
    from rl_stocks.runtime.colab.artifacts import validate_archive
    from rl_stocks.training.checkpoint_state import write_checkpoint_state
    def selected_step(m,f):
        m['selected_step']=512;f['metadata']['selected_step']=512
    _,plan,dataset=archive(tmp_path,mutate=selected_step)
    run=tmp_path/'artifacts/runs/fixture'
    (run/'best.zip').write_bytes(b'best512')
    (run/'checkpoint-1024.zip').write_bytes(b'final1024')
    (run/'selected-validation.json').write_text(json.dumps(result(.01)))
    state=write_checkpoint_state(run,result(.01),'checkpoint-1024.zip',1024,'post_update',512,'on_step',True)
    if tamper=='report':(run/'checkpoint-validation.json').write_text(json.dumps(result(.99)))
    if tamper=='final_flag':
        state['final_post_update']=False
        (run/'checkpoint-state.json').write_text(json.dumps(state))
    path=tmp_path/'checked.tar'
    with tarfile.open(path,'w') as t:t.add(tmp_path/'artifacts',arcname='artifacts')
    if tamper:
        with pytest.raises(RuntimeError):validate_archive(path,plan,dataset)
    else:assert len(validate_archive(path,plan,dataset))==1

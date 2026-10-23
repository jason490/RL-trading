"""Report routing and runtime approval are checked before any real evaluation."""
import importlib.util,json,sys
from pathlib import Path
import pytest
from test_recency_artifacts import profile

spec=importlib.util.spec_from_file_location('history_report',Path(__file__).resolve().parents[1]/'scripts/report_training_history.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('arm',['full','recent'])
@pytest.mark.parametrize('transport',[True,False])
def test_report_binds_arm_dataset_and_runtime_profile_through_payload(tmp_path,monkeypatch,arm,transport):
    identity=module.protocol()['datasets'][arm];approval=profile()
    root=tmp_path/'artifacts/colab/session';run=root/'retrieved/artifacts/runs/run';run.mkdir(parents=True)
    (run/'metadata.json').write_text(json.dumps(dict(seed=11,smoke=False,dataset=identity)))
    if transport:(root/'artifacts.tar.gz').write_bytes(b'fixture')
    state=tmp_path/'state.json';state.write_text(json.dumps(dict(status='complete',cleanup_verified=True,completed={'task':dict(session='session')})))
    universe=tmp_path/'configs/universes/sector_research.json';universe.parent.mkdir(parents=True);universe.write_text(json.dumps(dict(instruments=[])))
    rp=tmp_path/'profile.json';rp.write_text(json.dumps(approval))
    class ValidatedPayload(Exception):pass
    seen=[];payload=object()
    def validate(path,plan,dataset):
        assert dataset==identity and plan['dataset_contract']['identity']==identity
        assert plan['recency_study']['arm']==arm and plan['runtime_acceptance']==approval
        assert Path(path).exists();seen.append(arm)
    def validate_payload(data,actual_arm):
        assert data is payload and seen==[arm] and actual_arm==arm
        raise ValidatedPayload
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(module,'identify_arm',lambda m:arm)
    monkeypatch.setattr(module,'validate_archive',validate);monkeypatch.setattr(module,'read_dataset',lambda p:payload if p.name==identity else None)
    monkeypatch.setattr(module,'validate_payload',validate_payload)
    monkeypatch.setattr(sys,'argv',['report','--state',str(state),'--runtime-profile',str(rp),'--output',str(tmp_path/'summary.json')])
    with pytest.raises(ValidatedPayload):module.main()


def test_confirmation_job_preparation_requires_passing_screen_before_profile_or_allocation(tmp_path,monkeypatch):
    import runpy
    script=Path(__file__).resolve().parents[1]/'scripts/build_recency_jobs.py'
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(sys,'argv',[str(script),'--seed','23','--output','tasks.json'])
    with pytest.raises(SystemExit) as ex:runpy.run_path(str(script),run_name='__main__')
    assert ex.value.code==2 and not (tmp_path/'tasks.json').exists()

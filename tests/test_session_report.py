"""The matched report must validate each arm's own immutable dataset."""
import importlib.util,json,sys
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('session_report',Path(__file__).resolve().parents[1]/'scripts/report_session_context.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('arm',['legacy','masked','full'])
@pytest.mark.parametrize('transport',[True,False])
def test_report_binds_actual_arm_dataset_for_transport_or_retained_extraction(tmp_path,monkeypatch,arm,transport):
    contract=module.protocol();identity=contract['datasets'][arm]
    root=tmp_path/'artifacts/colab/session';run=root/'retrieved/artifacts/runs/run';run.mkdir(parents=True)
    (run/'metadata.json').write_text(json.dumps(dict(seed=11,smoke=False,dataset=identity)))
    if transport:(root/'artifacts.tar.gz').write_bytes(b'fixture')
    state=tmp_path/'state.json';state.write_text(json.dumps(dict(status='complete',cleanup_verified=True,completed={'task':dict(session='session')})))
    universe=tmp_path/'configs/universes/sector_research.json';universe.parent.mkdir(parents=True);universe.write_text(json.dumps(dict(instruments=[])))
    class ValidatedDataset(Exception):pass
    def validate(path,plan,dataset):
        assert dataset==identity and plan['dataset_contract']['identity']==identity
        assert Path(path).exists()
        raise ValidatedDataset
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(module,'identify_arm',lambda m:arm)
    monkeypatch.setattr(module,'validate_archive',validate)
    monkeypatch.setattr(sys,'argv',['report','--state',str(state),'--output',str(tmp_path/'summary.json')])
    with pytest.raises(ValidatedDataset):module.main()

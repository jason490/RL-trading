"""Accepted-study audit selection and source identity; synthetic reports only."""
import importlib.util,json
from copy import deepcopy
from pathlib import Path
import pytest
from test_action_execution import sample
from rl_stocks.inference.bundle import checksum
from rl_stocks.training.recurrent_cash import protocol

spec=importlib.util.spec_from_file_location('action_audit',Path(__file__).resolve().parents[1]/'scripts/diagnose_recurrent_actions.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def prepare(root,study="cash"):
    import importlib
    module_name={"cash":"recurrent_cash","alignment":"recurrent_alignment","recurrent-state":"recurrent_state","guard":"recurrent_guard","capacity":"recurrent_capacity","training-history":"training_history","l2":"recurrent_regularization"}[study]
    arms=importlib.import_module("rl_stocks.training."+module_name).protocol()["arms"]
    state=dict(status='complete',cleanup_verified=True,completed={});accepted={'cells':{}}
    for arm in arms:
        run=root/arm/'runs'/('run-'+arm);run.mkdir(parents=True)
        (run/'metadata.json').write_text(json.dumps(dict(status='complete',bundle_id=arm,dataset='fixture')))
        reports={}
        for phase,name in [('selected','selected-validation.json'),('final','validation.json')]:
            spy=sample();stock=deepcopy(spy);stock['symbol']='AAPL'
            # Include actual flat requests in the cash arm.
            if arm=='categorical_cash':
                spy['rows'][0]['raw_action'][0]=0;stock['rows'][0]['raw_action'][0]=0
            p=run/name;p.write_text(json.dumps({'spy':spy,'stock':stock}))
            reports[phase]=dict(file=name,sha256=checksum(p))
        state['completed'][arm]=dict(root=str(run.parents[1]))
        accepted['cells'][arm]=dict(run=run.name,bundle_id=arm,dataset='fixture',reports=reports)
    s=root/({'guard':'.runtime/guard-screen-v2-state.json','training-history':'.runtime/recency-screen-state.json'}.get(study,f'.runtime/{study}-screen-state.json'));s.parent.mkdir();s.write_text(json.dumps(state))
    name,date=module.STUDIES[study]
    p=root/f'results/{name}-{date}/summary.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(accepted))
    u=root/'configs/universes/sector_research.json';u.parent.mkdir(parents=True)
    u.write_text(json.dumps({'instruments':[dict(symbol='SPY',kind='etf',role='tradable'),dict(symbol='AAPL',kind='stock',role='tradable')]}))
    return s,p


def test_all_eight_phases_with_cash_are_audited_without_dataset_or_fitting(tmp_path,monkeypatch):
    prepare(tmp_path);monkeypatch.chdir(tmp_path)
    x=module.audit('cash')
    assert len(x['records'])==8 and x['source_files_unchanged']
    assert not any(x[k] for k in ['training_performed','dataset_opened','colab_allocated','test_evaluated'])
    cash=[r for r in x['records'] if r['arm']=='categorical_cash']
    assert all(r['groups']['stock']['means']['policy_zero_fraction']==1 for r in cash)
    assert len(x['source_files_sha256'])==15


@pytest.mark.parametrize('corruption',['cleanup','missing_arm','hash','metadata','extra_run','filename'])
def test_wrong_identity_or_incomplete_source_rejects(tmp_path,monkeypatch,corruption):
    sp,ap=prepare(tmp_path);s=json.loads(sp.read_text());a=json.loads(ap.read_text())
    arm=protocol()['arms'][0];run=Path(s['completed'][arm]['root'])/'runs'/a['cells'][arm]['run']
    if corruption=='cleanup':s['cleanup_verified']=False
    if corruption=='missing_arm':a['cells'].pop(arm)
    if corruption=='hash':(run/'validation.json').write_text('{}')
    if corruption=='metadata':(run/'metadata.json').write_text(json.dumps(dict(status='complete',bundle_id='other',dataset='fixture')))
    if corruption=='extra_run':
        p=run.parent/'extra';p.mkdir();(p/'metadata.json').write_text('{}')
    if corruption=='filename':a['cells'][arm]['reports']['final']['file']='../validation.json'
    sp.write_text(json.dumps(s));ap.write_text(json.dumps(a));monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):module.audit('cash')


@pytest.mark.parametrize('study',['recurrent-state','alignment'])
def test_existing_three_arm_registered_studies_remain_supported(tmp_path,monkeypatch,study):
    prepare(tmp_path,study);monkeypatch.chdir(tmp_path)
    result=module.audit(study)
    assert len(result['records'])==6 and result['source_files_unchanged']


def test_guard_audits_exact_v2_pair_without_opening_dataset(tmp_path,monkeypatch):
    prepare(tmp_path,'guard');monkeypatch.chdir(tmp_path)
    result=module.audit('guard')
    assert {(r['arm'],r['phase']) for r in result['records']}=={
        (a,p) for a in ['control','kl_guard'] for p in ['selected','final']}
    assert result['source_files_unchanged'] and not result['dataset_opened']
    assert '.runtime/guard-screen-v2-state.json' in result['source_files_sha256']


@pytest.mark.parametrize('study,arms',[('capacity',['control','larger']),('training-history',['full','recent']),('l2',['control','l2'])])
def test_exact_pair_and_changed_report_rejection(tmp_path,monkeypatch,study,arms):
    state_path,_=prepare(tmp_path,study);monkeypatch.chdir(tmp_path)
    result=module.audit(study)
    assert {(r['arm'],r['phase']) for r in result['records']}=={
        (a,p) for a in arms for p in ['selected','final']}
    assert result['source_files_unchanged'] and not result['dataset_opened']
    state=json.loads(state_path.read_text())
    root=Path(next(iter(state['completed'].values()))['root'])
    next((root/'runs').glob('*/validation.json')).write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):module.audit(study)

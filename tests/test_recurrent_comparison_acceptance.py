"""Orchestration fixtures only; physical checkpoints are verified in dedicated tests."""
from copy import deepcopy

import pytest

from test_recurrent_comparison import cells
from rl_stocks.core.contracts import digest
from rl_stocks.evaluation import recurrent_comparison_acceptance as acceptance
from rl_stocks.reporting.recurrent_comparison_report import render_comparison
from rl_stocks.training.recurrent_comparison import protocol,PROTOCOL_SHA256


def manifest(stage='screen'):
    seeds=[11] if stage=='screen' else [11,23,37]
    return dict(schema='recurrent-comparison-matrix-v1',protocol_sha256=PROTOCOL_SHA256,stage=stage,
        cells=[dict(arm=arm,seed=seed,folder=f'{arm}-{seed}',source_hash='a'*64,source_revision='b'*40)
               for seed in seeds for arm in protocol()['arms']])


def install_verifier(monkeypatch,*,candidate_score=.35,changed_seed11=False):
    calls=[]
    def verify(folder,dataset,starts,**expected):
        arm,seed=expected['expected_arm'],expected['expected_seed']
        calls.append((arm,seed))
        proof=deepcopy(cells(candidate_score)[arm])
        for phase in ('selected','final'):
            for group in ('stock','etf'):
                proof[phase]['groups'][group]['buy_hold_return']=.3
        for phase in ('selected','final'):
            proof[phase]['stock_excess_by_symbol']={symbol:proof[phase]['score']-.3 for symbol in
                set(protocol()['validation_symbols'])-set(protocol()['etf_symbols'])}
        proof.update(arm=arm,seed=seed,saved_artifacts_verified=True,
            artifact_inventory_sha256=digest(dict(arm=arm,seed=seed,changed=changed_seed11 and seed==11)),
            curve=[dict(steps=step,return_pct=10.+step/10000,change_pp=1.,excess_return_pp=-10.,
                        mean_max_drawdown_pct=-10.,buy_hold_return_pct=30.) for step in range(10000,100001,10000)])
        return proof
    monkeypatch.setattr(acceptance,'verify_cell',verify)
    return calls


def test_complete_screen_and_confirmation_reuse_original_screen_without_reselection(tmp_path,monkeypatch):
    calls=install_verifier(monkeypatch)
    screen=acceptance.accept_matrix(manifest(),{},tmp_path,root=tmp_path)
    assert len(calls)==3 and screen['gate']['passed'] and not screen['objective_achieved']
    assert screen['receipt_sha256']==acceptance.receipt_hash(screen)
    result=acceptance.accept_matrix(manifest('confirmation'),{},tmp_path,root=tmp_path,screen_receipt=screen)
    assert len(calls)==12 and result['gate']['passed'] and result['prospective_confirmation_required']
    assert not result['checkpoint_reselection'] and not result['retired_test_access']
    output=tmp_path/'report.html'
    render_comparison(result,output)
    text=output.read_text()
    assert 'plotly' in text and 'prospective evidence required' in text


def test_failed_screen_stays_failed_and_cannot_enter_confirmation(tmp_path,monkeypatch):
    calls=install_verifier(monkeypatch,candidate_score=.1)
    screen=acceptance.accept_matrix(manifest(),{},tmp_path,root=tmp_path)
    assert not screen['gate']['passed'] and len(calls)==3
    with pytest.raises(ValueError,match='passing original seed11'):
        acceptance.accept_matrix(manifest('confirmation'),{},tmp_path,root=tmp_path,screen_receipt=screen)
    assert len(calls)==3


def test_changed_seed11_or_tampered_receipt_rejects_before_other_seed_reads(tmp_path,monkeypatch):
    install_verifier(monkeypatch)
    screen=acceptance.accept_matrix(manifest(),{},tmp_path,root=tmp_path)
    bad=deepcopy(screen)
    bad['cells']['11']['gru_sac']['final']['score']=999.
    calls=install_verifier(monkeypatch)
    with pytest.raises(ValueError,match='unchanged passing'):
        acceptance.accept_matrix(manifest('confirmation'),{},tmp_path,root=tmp_path,screen_receipt=bad)
    assert not calls
    calls=install_verifier(monkeypatch,changed_seed11=True)
    with pytest.raises(ValueError,match='reused unchanged'):
        acceptance.accept_matrix(manifest('confirmation'),{},tmp_path,root=tmp_path,screen_receipt=screen)
    assert len(calls)==3 and all(seed==11 for _,seed in calls)


@pytest.mark.parametrize('change',['missing','duplicate','extra','bad_source','bool_seed'])
def test_incomplete_or_unbound_matrix_rejects_before_any_verification(tmp_path,monkeypatch,change):
    value=manifest()
    if change=='missing':
        value['cells'].pop()
    elif change=='duplicate':
        value['cells'][1]=value['cells'][0]
    elif change=='extra':
        value['cells'].append(dict(value['cells'][0],seed=23))
    elif change=='bad_source':
        value['cells'][0]['source_hash']='not a source hash'
    else:
        value['cells'][0]['seed']=True
    calls=install_verifier(monkeypatch)
    with pytest.raises(ValueError):
        acceptance.accept_matrix(value,{},tmp_path,root=tmp_path)
    assert not calls


def test_report_publication_failure_leaves_inputs_and_output_clean(tmp_path,monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    import sys
    spec=importlib.util.spec_from_file_location('accept_comparison_cli',Path(__file__).parents[1]/'scripts/accept_recurrent_comparison.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data=tmp_path/'data'
    data.mkdir()
    dataset=data/'dataset.json'
    dataset.write_text('{}')
    matrix=tmp_path/'matrix.json'
    matrix.write_text(json.dumps(manifest()))
    starts=tmp_path/'starts'
    starts.mkdir()
    output=tmp_path/'report'
    monkeypatch.setattr(sys,'argv',['accept','--matrix',str(matrix),'--dataset',str(dataset),
        '--starts',str(starts),'--output',str(output)])
    monkeypatch.setattr(module,'accept_matrix',lambda *args,**kwargs:dict(stage='screen',gate=dict(passed=False)))
    def failure(*args,**kwargs):
        raise RuntimeError('fixture render failure')
    monkeypatch.setattr(module,'render_comparison',failure)
    with pytest.raises(RuntimeError,match='render failure'):
        module.main()
    assert not output.exists() and not list(tmp_path.glob('.comparison-report-*'))
    assert dataset.read_text()=='{}' and json.loads(matrix.read_text())==manifest()

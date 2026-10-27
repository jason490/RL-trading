"""Read-only verification contracts, synthetic curves, actual untrained checkpoints."""
from copy import deepcopy

import pytest
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
import torch

from rl_stocks.reporting.recurrent_sac_comparison import phase_summary
from rl_stocks.training.recurrent_comparison import protocol
from rl_stocks.training.recurrent_comparison_starts import construct
from rl_stocks.training.recurrent_comparison_worker import NativeAudit,native_snapshot
from rl_stocks.training.recurrent_comparison_verify import verify_history,verify_native_checkpoint
from rl_stocks.training.sac_update_worker import preserved_rng


def curves():
    history=[dict(steps=step,score=.2 if step in (20000,30000) else .1,phase='post_update',
                  learner_updates=step-1000,checkpoint=f'checkpoint-{step}',
                  training=dict(native_library_counter=step-1000,
                    optimizer_steps=dict.fromkeys(('actor','critic','entropy'),step-1000),counter_kind='gradient_updates')) for step in range(10000,100001,10000)]
    def pointer(row):
        return dict(steps=row['steps'],score=row['score'],updates=row['learner_updates'],checkpoint=row['checkpoint'])
    return history,pointer(history[1]),pointer(history[-1])


def test_earliest_tie_selection_final_distinct_and_bad_counter_or_phase_rejects():
    history,selected,final=curves()
    assert verify_history(history,selected,final,'native_sac')['steps']==20000
    for change in ('tie','final','phase','count','nan','kind','missing'):
        rows,s,f=deepcopy((history,selected,final))
        if change=='tie':
            s.update(steps=30000,updates=29000,checkpoint='checkpoint-30000')
        elif change=='final':
            f=s
        elif change=='phase':
            rows[0]['phase']='on_step'
        elif change=='count':
            rows[0]['learner_updates']=10000
        elif change=='nan':
            rows[0]['score']=float('nan')
        elif change=='kind':
            rows[0]['training']['counter_kind']='ppo_epochs'
        else:
            rows.pop()
        with pytest.raises(ValueError):
            verify_history(rows,s,f,'native_sac')


@pytest.mark.parametrize('arm',['native_sac','recurrent_ppo_reference'])
def test_actual_zero_update_native_checkpoint_accepts_then_checksum_rejects(tmp_path,monkeypatch,arm):
    def forbidden(*args,**kwargs):
        pytest.fail('local fitting attempted')
    for cls in (SAC,RecurrentPPO):
        monkeypatch.setattr(cls,'train',forbidden)
        monkeypatch.setattr(cls,'learn',forbidden)
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
    with preserved_rng():
        model=construct(arm,11)
        audit=NativeAudit(model,arm)
        lineage=dict(source='untrained-only-fixture')
        try:
            native_snapshot(model,arm,11,0,audit,tmp_path,lineage)
            restored=verify_native_checkpoint(tmp_path/'initial',arm=arm,seed=11,steps=0,lineage=lineage)
            assert restored._n_updates==0
            path=tmp_path/'initial/model.zip'
            path.write_bytes(path.read_bytes()+b'corruption')
            with pytest.raises(ValueError,match='checksum'):
                verify_native_checkpoint(tmp_path/'initial',arm=arm,seed=11,steps=0,lineage=lineage)
        finally:
            audit.close()


def test_exact_panel_summary_and_future_decliners_reporting_only():
    spec=protocol()
    results={symbol:dict(symbol=symbol,metrics=dict(net_return=.1,max_drawdown=-.2),
         baselines=dict(buy_hold=dict(metrics=dict(net_return=.05,max_drawdown=-.1))),
         directional=dict(short_net_pnl=0.)) for symbol in spec['validation_symbols']}
    # Both must be positive before describing a declining-stock episode as profitable shorting.
    results['AAPL']['baselines']['buy_hold']['metrics']['net_return']=-.1
    results['AAPL']['directional']['short_net_pnl']=100.
    results['MSFT']['metrics']['net_return']=-.1
    results['MSFT']['baselines']['buy_hold']['metrics']['net_return']=-.2
    results['MSFT']['directional']['short_net_pnl']=100.
    summary=phase_summary(results)
    assert summary['groups']['stock']['count']==22 and summary['groups']['etf']['count']==4
    decline=summary['declining_stocks_reporting_only']
    assert decline['count']==2 and sum(r['positive_policy_and_short_pnl'] for r in decline['records'])==1
    changed=deepcopy(results)
    changed['AAPL']['directional']['short_net_pnl']=-100
    altered=phase_summary(changed)
    assert altered['score']==summary['score'] and altered['groups']==summary['groups']
    del results['DIA']
    with pytest.raises(ValueError,match='exact one-episode'):
        phase_summary(results)


def test_effective_native_configuration_rejects_real_changes_and_nonconstant_schedule():
    from stable_baselines3.common.utils import FloatSchedule,LinearSchedule
    from rl_stocks.training.recurrent_comparison import validate_native_parameters
    with preserved_rng():
        model=construct('native_sac',11)
        validate_native_parameters(model,'native_sac')
        model.target_entropy=-3.
        with pytest.raises(ValueError,match='numeric parameter'):
            validate_native_parameters(model,'native_sac')
        model.target_entropy=-4.
        model.policy_kwargs['use_sde']=True
        with pytest.raises(ValueError,match='policy_kwargs'):
            validate_native_parameters(model,'native_sac')
        ppo=construct('recurrent_ppo_reference',11)
        validate_native_parameters(ppo,'recurrent_ppo_reference')
        ppo.policy.optimizer.param_groups[0]['lr']=.01
        with pytest.raises(ValueError,match='learning rate'):
            validate_native_parameters(ppo,'recurrent_ppo_reference')
        ppo.policy.optimizer.param_groups[0]['lr']=.0003
        ppo.clip_range=FloatSchedule(LinearSchedule(.2,.1,.5))
        with pytest.raises(ValueError,match='constant schedule'):
            validate_native_parameters(ppo,'recurrent_ppo_reference')


def test_shared_economic_replay_rejects_action_or_cash_drift_without_policy_rerun():
    from test_recurrent_sac_evaluation import fixture
    from rl_stocks.evaluation.recurrent_sac import evaluate_recurrent_sac
    from rl_stocks.evaluation.recorded_economics import verify_report
    networks,dataset=fixture()
    results=evaluate_recurrent_sac(networks,dataset,exposure_mode='unit_magnitude')
    verify_report(results,dataset)  # same economics path used for native controls
    verify_report(results,dataset,networks)  # optional local frozen/direct GRU parity
    bad=deepcopy(results)
    first=next(iter(bad.values()))
    first['rows'][0]['raw_action']=(.5,*first['rows'][0]['raw_action'][1:])
    with pytest.raises(ValueError,match='action contract'):
        verify_report(bad,dataset)
    cash_bad=deepcopy(results)
    next(iter(cash_bad.values()))['rows'][0]['cash']+=1
    with pytest.raises(ValueError,match='saved development economics'):
        verify_report(cash_bad,dataset)

from copy import deepcopy
import runpy
import pytest
from rl_stocks.core.contracts import digest


def inputs():
    summaries=[];protocols=[]
    for index,seed in enumerate((11,23,37)):
        protocol={'source_seed':seed,'training_source_revision':str(seed)};protocols.append(protocol)
        cells=[]
        for phase in ('selected','final'):
            for mode,score in (('unrestricted',.1+.01*index),('long_only',.12+.01*index),('short_only',-.02+.01*index)):
                groups={g:{'net_return':score,'buy_hold_return':.3,'excess_return':score-.3,
                    'max_drawdown':-.2,'buy_hold_max_drawdown':-.3,'median_excess_return':score-.3} for g in ('stock','etf')}
                cohort=[]
                for symbol in ('AMZN','DIS','NEE'):
                    positive=mode=='short_only' and (seed!=37 or symbol=='NEE')
                    cohort.append({'symbol':symbol,'policy_return':.05 if positive else -.05,
                        'buy_hold_return':-.2,'short_net_pnl':1000 if positive else -1000,
                        'positive_policy_and_short_pnl':positive})
                cells.append({'profile':'legacy','phase':phase,'mode':mode,'weights_sha256':str(seed)+phase,
                    'metrics':{'score':score,'groups':groups},'mean_costs':100.,
                    'declining_stocks':{'records':cohort},'unrestricted_parity':{'passed':True,'quantity_path_exact':seed!=37}})
        summaries.append({'status':'complete','source_artifacts_unchanged':True,'training_performed':False,
            'test_evaluated':False,'colab_allocated':False,'protocol_sha256':digest(protocol),
            'source_run_contract':{'dataset':'fixture','source_hash':str(seed)},
            'validation_baseline_sha256':'same','source':{'revision':'fixture'},'cells':cells})
    return summaries,protocols


def test_paired_three_seed_report_keeps_unsuccessful_modes_and_short_cohort():
    aggregate=runpy.run_path('scripts/report_direction_replay.py')['aggregate']
    summaries,protocols=inputs();result=aggregate(summaries,protocols)
    final={r['mode']:r for r in result['results'] if r['phase']=='final'}
    assert final['unrestricted']['mean_return']==pytest.approx(.11)
    assert final['long_only']['mean_gain_vs_paired_unrestricted_pp']==pytest.approx(2.)
    assert final['short_only']['mean_return']==pytest.approx(-.01)
    assert final['short_only']['positive_total_and_short_count']==7
    assert final['short_only']['declining_stock_seed_observations']==9
    assert not result['objective_achieved'] and result['seed37_selected_bounded_reference_exception']


def test_paired_report_rejects_missing_or_unmatched_evidence():
    aggregate=runpy.run_path('scripts/report_direction_replay.py')['aggregate']
    summaries,protocols=inputs()
    for mutate in (
        lambda s:s[0].update(status='calibrated'),
        lambda s:s[1].update(validation_baseline_sha256='different'),
        lambda s:s[1]['source_run_contract'].update(dataset='different'),
        lambda s:s[1]['cells'][1].update(weights_sha256='different'),
        lambda s:s[2]['cells'].pop(),
        lambda s:s[2]['cells'][0]['unrestricted_parity'].update(passed=False),
        lambda s:s[1]['cells'][0]['declining_stocks']['records'][0].update(positive_policy_and_short_pnl=True),
    ):
        changed=deepcopy(summaries);mutate(changed)
        with pytest.raises(ValueError):aggregate(changed,protocols)
    with pytest.raises(ValueError):aggregate(summaries[:2],protocols[:2])


def test_feedback_compares_original_requests_instead_of_mechanical_filter_output():
    feedback=runpy.run_path('scripts/report_direction_replay.py')['action_feedback']
    source={'stock':{'rows':[{'session':'2021-01-01','raw_action':[.5,1,1,1]}]}}
    same={'stock':{'rows':[{'session':'2021-01-01','raw_action':[0.,1,1,1],'policy_action':[.5,1,1,1]}]}}
    assert feedback(source,same)['requested_direction_sign_flips']==0
    assert feedback(source,same)['mean_absolute_requested_exposure_change']==0
    changed=deepcopy(same);changed['stock']['rows'][0]['policy_action'][0]=-.25
    stats=feedback(source,changed)
    assert stats['requested_direction_sign_flips']==1
    assert stats['mean_absolute_requested_exposure_change']==.75
    with pytest.raises(ValueError):feedback(source,{})

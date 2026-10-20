from copy import deepcopy
from pathlib import Path
import pytest
from rl_stocks.runtime.colab.jobs import context_plan,validate_plan,validate_context_payload,CONTEXT_DATASETS
from rl_stocks.runtime.colab.fleet import validate_tasks
from rl_stocks.features.registry.assemble import NAMES as BASE
from rl_stocks.features.trend.context import NAMES as EXTRA


def payload(mode):
    return {'schema':'daily-indicators-v1-trend-context-v1-'+mode,
        'names':list(BASE)+(list(EXTRA) if mode!='matched_baseline' else []),
        'manifest':{'trend_context':{'mode':mode},'included_splits':['train','validation'],'unverified_override':False},
        'episodes':{'fixture':{'split':'train','bars':[{'session':'2020-01-02'}]}}}


def test_context_plans_lock_data_schema_hardware_and_every_job_setting():
    for arm in CONTEXT_DATASETS:
        for smoke in (True,False):
            p=context_plan(arm,smoke=smoke);assert validate_plan(p)==p
            validate_context_payload(payload(arm),p)
            for mutate in [lambda v:v['dataset_contract'].update(identity='different'),
                lambda v:v['jobs'][0].update(reward_kind='net_initial_equity_change'),
                lambda v:v.update(accelerator='T4'),lambda v:v['jobs'][0].update(steps=250000)]:
                changed=deepcopy(p);mutate(changed)
                with pytest.raises(ValueError):validate_plan(changed)
    for arm,seed,smoke in [('unknown',11,False),('full',23,True),('full',True,False)]:
        with pytest.raises(ValueError):context_plan(arm,seed,smoke)


def test_payload_rejects_mismatched_inputs_unverified_data_and_test_bars():
    p=context_plan('full')
    for mutate in [lambda d:d['names'].reverse(),lambda d:d.update(schema='wrong'),
        lambda d:d['manifest']['trend_context'].update(mode='masked'),
        lambda d:d['manifest'].update(included_splits=['train','validation','test']),
        lambda d:d['manifest'].update(unverified_override=True),
        lambda d:d['episodes']['fixture'].update(split='test'),
        lambda d:d['episodes']['fixture']['bars'][0].update(session='2024-01-02')]:
        d=payload('full');mutate(d)
        with pytest.raises(ValueError,match='dataset contract'):validate_context_payload(d,p)


def test_wrong_dataset_task_rejected_before_allocation(tmp_path):
    p=tmp_path/'wrong';p.mkdir();(p/'dataset.json').write_text('{}')
    with pytest.raises(ValueError,match='dataset identity'):
        validate_tasks([{'id':'full-11','dataset':str(p),'plan':context_plan('full')}],tmp_path)

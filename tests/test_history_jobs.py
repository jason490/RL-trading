from copy import deepcopy
import pytest
from rl_stocks.runtime.colab.jobs import history_plan,validate_plan
from rl_stocks.training.history.spec import protocol,validate_payload,validate_counts
from rl_stocks.runtime.colab.fleet import validate_tasks


def test_fixed_history_plan_rejects_every_mutation():
    for arm in protocol()['arms']:
        for smoke in (True,False):
            p=history_plan(arm,smoke=smoke);assert validate_plan(p)==p
            for mutate in [lambda v:v.update(protocol_sha256='wrong'),lambda v:v['jobs'][0].update(start_sampling='legacy_uniform'),lambda v:v['jobs'][0].update(steps=250000),lambda v:v.update(accelerator='T4'),lambda v:v['dataset_contract'].update(identity='wrong')]:
                bad=deepcopy(p);mutate(bad)
                with pytest.raises(ValueError):validate_plan(bad)
    for args in [('unknown',11,False),('original',23,True),('original',True,False)]:
        with pytest.raises(ValueError):history_plan(*args)


def test_wrong_data_rejected_before_allocation(tmp_path):
    folder=tmp_path/'wrong';folder.mkdir();(folder/'dataset.json').write_text('{}')
    with pytest.raises(ValueError,match='identity'):validate_tasks([dict(id='original',dataset=str(folder),plan=history_plan('original'))],tmp_path)
    with pytest.raises(ValueError,match='identity'):validate_payload({},'original')


def counts():return dict(mode='full_window_uniform',downtrend_counts_measured=False,transitions=512,starts=5,downtrend_starts=0,downtrend_transitions=0,history_groups={'SPY:recent':dict(starts=5,transitions=512)})


def test_group_counts_accept_empty_final_reset_and_reject_mismatches():
    assert validate_counts(counts(),'original',True,512)['window_transitions']==128
    for mutate in [lambda c:c.update(transitions=511),lambda c:c.update(starts=4),lambda c:c.update(downtrend_counts_measured=True),lambda c:c['history_groups']['SPY:recent'].update(transitions=513),lambda c:c.update(history_groups={'SPY:earlier':dict(starts=5,transitions=512)})]:
        c=counts();mutate(c)
        with pytest.raises(ValueError):validate_counts(c,'original',True,512)
    c=counts();c['history_groups']={'SPY:earlier':dict(starts=5,transitions=512)}
    validate_counts(c,'older_etf',True,512)

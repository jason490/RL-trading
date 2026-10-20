from copy import deepcopy
import numpy as np
import pytest
from rl_stocks.runtime.colab.jobs import diversity_plan,validate_plan
from rl_stocks.training.diversity import protocol,validate_payload,validate_counts
from rl_stocks.environment.sampling.windows import StockEtfCycles


def counts(arm,smoke):
    p=protocol();window=128 if smoke else 512;steps=1024 if smoke else 100000
    # Sampling only, using source-ready dates; no environment fitting.
    from datetime import date,timedelta
    bars=[dict(session=str(date(2010,1,1)+timedelta(days=i))) for i in range(700)]
    episodes=[dict(symbol=s,split='train',bars=bars) for s in p['training_symbols'][arm]]
    sampler=StockEtfCycles(episodes,window);rng=np.random.default_rng(11)
    groups={};remaining=steps;starts=0
    while remaining:
        index,_=sampler.choose(rng);n=min(remaining,window);remaining-=n;starts+=1
        r=groups.setdefault(sampler.groups[index],dict(starts=0,transitions=0));r['starts']+=1;r['transitions']+=n
    return dict(mode=sampler.mode,starts=starts,transitions=steps,downtrend_starts=0,downtrend_transitions=0,
                downtrend_counts_measured=False,history_groups=groups)


def test_fixed_diversity_contracts_and_actual_cycles():
    for arm in protocol()['arms']:
        for smoke in (True,False):
            plan=diversity_plan(arm,smoke=smoke);assert validate_plan(plan)==plan
            checked=validate_counts(counts(arm,smoke),arm,smoke,plan['jobs'][0]['steps'])
            if not smoke:assert checked['symbols_with_transitions']==len(protocol()['training_symbols'][arm])
            for change in (lambda p:p['jobs'][0].update(steps=250000),lambda p:p['jobs'][0].update(start_sampling='legacy_uniform'),lambda p:p['jobs'][0].update(overrides={'ent_coef':'auto_0.001'})):
                bad=deepcopy(plan);change(bad)
                with pytest.raises(ValueError):validate_plan(bad)
    with pytest.raises(ValueError):validate_payload({},'expanded')


def test_counts_reject_missing_coverage_and_false_totals():
    c=counts('expanded',False);c['history_groups'].pop(next(iter(c['history_groups'])))
    with pytest.raises(ValueError):validate_counts(c,'expanded',False,100000)
    c=counts('original',True);c['transitions']+=1
    with pytest.raises(ValueError):validate_counts(c,'original',True,1024)

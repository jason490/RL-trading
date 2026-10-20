from copy import deepcopy
from datetime import date,timedelta
import pytest
from rl_stocks.training.diversity import audit_sources,fixed_preprocessing_expansion,ETFS


def sources():
    dates=[(date(2010,1,1)+timedelta(days=i)).isoformat() for i in range(513)]
    episodes={}
    for symbol in (*ETFS,'AAA'):
        for split,days in [('train',dates),('validation',['2021-01-04','2021-01-05'])]:
            episodes[symbol+'-'+split]=dict(symbol=symbol,split=split,bars=[dict(symbol=symbol,session=d) for d in days],
                                           features=[[1.] for _ in days],atr=[1. for _ in days])
    annotations=[dict(symbol=s,kind='broad_etf' if s in ETFS else 'stock',sector='diversified' if s in ETFS else 'technology') for s in (*ETFS,'AAA')]
    original=dict(schema='fixture',names=['f'],normalizer=dict(names=['f'],means=[0.],scales=[1.]),episodes=episodes,
                  manifest=dict(included_splits=['train','validation'],unverified_override=False,purge_sessions=20,
                                max_holding_sessions=20,splits=dict(train_end='2020-12-31'),
                                evaluation_symbols=list(ETFS)+['AAA'],universe_config=dict(instruments=annotations)))
    expanded=deepcopy(original);expanded['normalizer']['means']=[2.]
    extra=deepcopy(episodes['AAA-train']);extra['symbol']='BBB'
    for b in extra['bars']:b['symbol']='BBB'
    expanded['episodes']['BBB-train']=extra
    expanded['manifest']['universe_config']['instruments'].append(dict(symbol='BBB',kind='stock',sector='technology'))
    return original,expanded


def test_original_episodes_and_preprocessing_unchanged_after_expansion():
    original,expanded=sources();before=deepcopy((original,expanded))
    audit=audit_sources(original,expanded);result=fixed_preprocessing_expansion(original,expanded)
    assert (original,expanded)==before and result['normalizer']==original['normalizer']
    assert not audit['normalizers_equal'] and audit['additional_training_symbols']==['BBB']
    assert result['episodes']==expanded['episodes'] and result['manifest']['sac_diversity']['normalizer_sha256']==audit['original_normalizer_sha256']
    result['normalizer']['means'][0]=999
    assert original['normalizer']['means']==[0.] and expanded['normalizer']['means']==[2.]


@pytest.mark.parametrize('change',[
    lambda x:x['episodes']['AAA-train']['features'][0].__setitem__(0,99.),
    lambda x:x['episodes']['AAA-validation']['bars'][0].update(session='2024-01-02'),
    lambda x:x['episodes']['BBB-train']['atr'].pop(),
    lambda x:x['episodes']['BBB-train'].update(split='test'),
    lambda x:x['manifest'].update(unverified_override=True),
    lambda x:x['episodes'].pop('SPY-train'),
    lambda x:x['manifest']['evaluation_symbols'].append('BBB'),
    lambda x:x['episodes']['BBB-train']['bars'][0].update(symbol='AAA'),
])
def test_unmatched_or_invalid_expansion_rejected(change):
    a,b=sources();change(b)
    with pytest.raises(ValueError):audit_sources(a,b)

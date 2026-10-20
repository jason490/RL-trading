from copy import deepcopy
import pytest
from rl_stocks.evaluation.training_fit import training_window
from rl_stocks.evaluation.runner import evaluate


def data():
    dates = ['2017-12-28','2017-12-29','2018-01-02','2020-12-02','2020-12-03']
    train = dict(symbol='SPY',split='train',bars=[dict(session=d) for d in dates],
                 features=[[i] for i in range(5)],atr=list(range(1,6)))
    return dict(manifest=dict(included_splits=['train','validation'],unverified_override=False,
                             evaluation_symbols=['SPY']),normalizer={'fixed':True},
                episodes={'train':train,'development':dict(symbol='SPY',split='validation',bars=[dict(session='2021-01-04')])})


def test_exact_alignment_anchor_and_original_labels_unchanged():
    source=data();before=deepcopy(source);window=training_window(source)
    assert source==before and set(window['episodes'])=={'train'}
    e=window['episodes']['train']
    assert e['split']=='train' and e['features']==[[1],[2],[3]] and e['atr']==[2,3,4]
    assert window['normalizer']==source['normalizer']


@pytest.mark.parametrize('mutate',[
    lambda x:x['episodes']['train']['bars'].pop(2),
    lambda x:x['episodes']['train']['bars'][0].update(session='2024-01-02'),
    lambda x:x['episodes']['train']['features'].pop(),
    lambda x:x['episodes']['development'].update(split='test'),
    lambda x:x['manifest'].update(unverified_override=True),
    lambda x:x['manifest']['evaluation_symbols'].append('QQQ'),
])
def test_invalid_source_and_window_rejected(mutate):
    x=data();mutate(x)
    with pytest.raises(ValueError):training_window(x)


def test_standard_evaluator_still_rejects_training_split():
    with pytest.raises(ValueError,match='evaluation split'):
        evaluate(None,{},split='train')


def test_real_simulation_keeps_train_labels_and_next_session_anchor():
    from dataclasses import asdict
    from test_core import bar
    import numpy as np
    from rl_stocks.evaluation.training_fit import evaluate_training_window
    from rl_stocks.environment.execution.simulator import SimulationConfig
    source=data()
    e=source['episodes']['train']
    e['bars']=[asdict(bar(d)) for d in ['2017-12-28','2017-12-29','2018-01-02','2020-12-02','2020-12-03']]
    source.update(names=['fixture'],normalizer=dict(names=['fixture'],means=[0.],scales=[1.]))
    class Flat:
        def predict(self,observation,deterministic=True):
            assert deterministic and len(observation)==15
            return np.zeros(4,dtype=np.float32),None
    before=deepcopy(source)
    result=evaluate_training_window(Flat(),training_window(source),SimulationConfig(),'execution_full')['train']
    assert source==before and result['split']=='train'
    assert [r['session'] for r in result['rows']]==['2018-01-02','2020-12-02']
    assert result['metrics']['net_return']==0 and result['directional']['short_net_pnl']==0

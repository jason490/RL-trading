from copy import deepcopy
import numpy as np
import pytest
from test_exposure_replay import fixture
from rl_stocks.inference.ensemble import UniformEnsemble
from rl_stocks.evaluation.runner import evaluate


class Member:
    def __init__(self,identity,action):
        self.identity=identity;self.weights_sha256=identity;self.contract={'observation_size':6,'feed':'fixture'}
        self.output=np.array(action,dtype=np.float32);self.seen=[]
    def action(self,observation):self.seen.append(observation.copy());return self.output.copy()


def test_uniform_all_four_components_preserves_members_and_shared_input():
    a=Member('a',[.8,-.2,.4,.6]);b=Member('b',[-.4,.8,-.2,1.]);ensemble=UniformEnsemble([a,b])
    obs=np.arange(6,dtype=np.float32);before=obs.copy()
    action,_=ensemble.predict(obs)
    np.testing.assert_allclose(action,[.2,.3,.1,.8],atol=1e-7)
    np.testing.assert_array_equal(obs,before)
    np.testing.assert_array_equal(a.seen[0],b.seen[0])
    assert [r['weight'] for r in ensemble.recipe()['members']]==[.5,.5]
    np.testing.assert_array_equal(action,ensemble.predict(obs)[0])


def test_ensemble_replays_a_single_account_instead_of_averaging_returns():
    a=Member('long',[.7,1,1,1]);b=Member('short',[-.7,1,1,1]);ensemble=UniformEnsemble([a,b])
    r=evaluate(ensemble,fixture())['SPY/validation']
    assert r['metrics']['net_return']==0 and r['metrics']['costs']==0
    for left,right in zip(a.seen,b.seen):np.testing.assert_array_equal(left,right)
    # Flat aggregate exposure feeds back to BOTH policies each session.
    assert all(x[1]==0 and x[2]==1 for x in a.seen)


def test_ensemble_rejects_duplicate_incompatible_stochastic_or_invalid_contracts():
    a=Member('a',[0,1,1,1]);b=Member('b',[0,1,1,1])
    with pytest.raises(ValueError):UniformEnsemble([])
    with pytest.raises(ValueError):UniformEnsemble([a,a])
    b.contract['feed']='other'
    with pytest.raises(ValueError):UniformEnsemble([a,b])
    e=UniformEnsemble([a])
    for bad in ([0]*5,[np.nan]*6):
        with pytest.raises(ValueError):e.predict(bad)
    with pytest.raises(ValueError):e.predict([0]*6,deterministic=False)
    for bad in ([0]*3,[np.nan,0,0,0],[1.01,0,0,0]):
        a.output=np.array(bad)
        with pytest.raises(ValueError):e.predict([0]*6)


def test_recipe_roundtrip_rejects_changed_weights_members_or_normalization():
    import json
    a,b=Member('a',[.5,1,0,-1]),Member('b',[-.5,-1,0,1]);original=UniformEnsemble([a,b])
    recipe=json.loads(json.dumps(original.recipe()));restored=UniformEnsemble.from_recipe(recipe,{'a':a,'b':b})
    np.testing.assert_array_equal(original.predict([0]*6)[0],restored.predict([0]*6)[0])
    for mutate in (lambda r:r['members'][0].update(weight=.9),lambda r:r['members'][0].update(weights_sha256='other'),lambda r:r['contract'].update(feed='other')):
        bad=deepcopy(recipe);mutate(bad)
        with pytest.raises(ValueError):UniformEnsemble.from_recipe(bad,{'a':a,'b':b})
    with pytest.raises(ValueError):UniformEnsemble.from_recipe(recipe,{'a':a})


def test_fixed_saved_member_matches_sb3_prediction_and_validates_spaces(tmp_path):
    from stable_baselines3 import PPO
    from rl_stocks.environment.trading.env import TradingEnv
    from rl_stocks.data.generation.datasets import normalizer_from
    from rl_stocks.inference.ensemble import FixedMember
    from rl_stocks.inference.bundle import checksum
    d=fixture();env=TradingEnv(list(d['episodes'].values()),normalizer_from(d['normalizer']),d['names'])
    model=PPO('MlpPolicy',env,n_steps=8,batch_size=8,seed=11,device='cpu');model.save(tmp_path/'weights')
    loaded=PPO.load(tmp_path/'weights.zip',device='cpu');contract={'observation_size':6,'feed':'fixture'}
    member=FixedMember(loaded,contract,'saved',checksum(tmp_path/'weights.zip'));ensemble=UniformEnsemble([member])
    obs,_=env.reset(seed=11)
    np.testing.assert_array_equal(ensemble.predict(obs)[0],model.predict(obs,deterministic=True)[0])
    loaded.action_space.high[:]=2
    with pytest.raises(ValueError):FixedMember(loaded,contract,'bad','hash')


def test_retired_source_dataset_cannot_be_selected_for_evaluation(tmp_path):
    import json,runpy
    from pathlib import Path
    prepare=runpy.run_path('scripts/replay_ensembles.py')['prepare']
    protocol=json.loads(Path('configs/experiments/frozen_ensembles.json').read_text())
    protocol['evaluation_dataset']=protocol['source_dataset']
    with pytest.raises(ValueError,match='before data access'):
        prepare(protocol,{'status':'complete','cleanup_verified':True},tmp_path/'does-not-exist')


def test_resume_rejects_changed_models_data_or_gates_and_tampered_cached_paths(tmp_path):
    import json,gzip,runpy
    from pathlib import Path
    from rl_stocks.core.contracts import canonical,digest
    from rl_stocks.inference.bundle import checksum
    restore=runpy.run_path('scripts/replay_ensembles.py')['verified_cache']
    protocol=json.loads(Path('configs/experiments/frozen_ensembles_continuation.json').read_text())
    original=json.loads(Path('configs/experiments/frozen_ensembles.json').read_text())
    path=tmp_path/'report.json.gz';path.write_bytes(gzip.compress(canonical({'fixture':True}),mtime=0))
    proof={'fixture':'contract'}
    old={'status':'failed','training_performed':False,'colab_allocated':False,'test_evaluated':False,
         'package_source_hash':'fixed','preflight':proof,'protocol_sha256':digest(original),'protocol':original,
         'failed_case':{'name':'reference:sac-11','phase':'final','source_reference_drift':{**{k:0. for k in original['reference_drift_limits']},'maximum_cost_difference_dollars':3.24}},
         'cells':[{'phase':'selected','name':'ppo_mean','report':path.name,'report_sha256':checksum(path)}]}
    (tmp_path/'summary.json').write_text(json.dumps(old))
    cache,_=restore(tmp_path,protocol,proof,'fixed');assert cache['selected','ppo_mean'][1]['fixture']
    for changed in ({**protocol,'candidates':{}},{**protocol,'evaluation_dataset':'other'}):
        with pytest.raises(ValueError):restore(tmp_path,changed,proof,'fixed')
    with pytest.raises(ValueError):restore(tmp_path,protocol,{'other':'contract'},'fixed')
    with pytest.raises(ValueError):restore(tmp_path,protocol,proof,'changed-code')
    path.write_bytes(b'tampered')
    with pytest.raises(ValueError):restore(tmp_path,protocol,proof,'fixed')

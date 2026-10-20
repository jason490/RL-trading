from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from rl_stocks.inference.forest_arrays import validate_tree,export_forest,load_forest,predict_forest


def tree():
    return {'children_left':np.array([1,-1,-1]),'children_right':np.array([2,-1,-1]),
        'feature':np.array([0,-2,-2]),'threshold':np.array([0.,-2.,-2.]),'value':np.array([0.,-.02,.03])}


def test_portable_thresholds_mean_and_safe_roundtrip_without_fitting(tmp_path):
    t=validate_tree(tree());x=np.zeros((3,10));x[:,0]=[-1,0,1]
    np.testing.assert_array_equal(predict_forest([t,t],x),[-.02,-.02,.03])
    stub=SimpleNamespace(n_features_in_=10,n_outputs_=1,estimators_=[SimpleNamespace(tree_=SimpleNamespace(**tree()))])
    path=tmp_path/'forest.npz';export_forest(stub,path);restored=load_forest(path,1)
    np.testing.assert_array_equal(predict_forest(restored,x),predict_forest([t],x))
    with pytest.raises(ValueError):load_forest(path,2)
    for bad in [np.full((1,10),float('nan')),np.ones((1,9)),np.full((1,10),1e300)]:
        with pytest.raises(ValueError):predict_forest(restored,bad)


def test_bad_tree_structure_and_parameters_rejected():
    for mutate in [lambda t:t['children_left'].__setitem__(0,0),lambda t:t['children_right'].__setitem__(1,0),
        lambda t:t['feature'].__setitem__(0,10),lambda t:t['threshold'].__setitem__(0,float('nan')),
        lambda t:t.update(feature=np.array([0.,-2.,-2.])),lambda t:t['children_right'].__setitem__(0,100)]:
        t=deepcopy(tree());mutate(t)
        with pytest.raises(ValueError):validate_tree(t)


def test_optional_constructor_matches_fixed_tree_budget_without_fitting():
    pytest.importorskip('sklearn')
    from rl_stocks.training.supervised.forest_spec import make_forest,verify_forest_environment
    model=make_forest(23);assert model.n_estimators==128 and model.max_depth==8 and model.n_jobs==1
    assert not hasattr(model,'estimators_') and make_forest(11,True).n_estimators==2
    assert verify_forest_environment()['trees']['scikit-learn']=='1.9.1'
    with pytest.raises(ValueError):make_forest(23,True)
    with pytest.raises(ValueError):make_forest(True)

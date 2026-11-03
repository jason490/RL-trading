from copy import deepcopy

import numpy as np
import pytest

from rl_stocks.inference.boosting_arrays import (
    load_boosting,
    predict_boosting,
    save_boosting,
)
from test_forest_arrays import tree


def test_additive_residual_ridge_float32_threshold_and_immutable_roundtrip(tmp_path):
    path = tmp_path / "boost.npz"
    ridge = np.array([0.1, 0.2] + [0.0] * 9)
    original = ridge.copy()
    save_boosting(path, ridge, [tree()] * 256)
    model = load_boosting(path)
    x = np.zeros((3, 10))
    x[:, 0] = [-1.0, 0.0, 1.0]
    wanted = 0.1 + 0.2 * x[:, 0] + 256 * 0.03 * np.array([-0.02, -0.02, 0.03])
    np.testing.assert_allclose(predict_boosting(model, x), wanted, atol=1e-14)
    np.testing.assert_array_equal(predict_boosting(model, x), predict_boosting(model, x))
    np.testing.assert_array_equal(ridge, original)
    for array in [model.ridge, *model.trees[0].values()]:
        with pytest.raises(ValueError):
            array[0] = 1
    with pytest.raises(TypeError):
        model.trees[0]["value"] = np.zeros(3)
    with pytest.raises(ValueError):
        model.ridge.flags.writeable = True
    boundary = np.zeros((1, 10))
    boundary[0, 0] = np.nextafter(0.0, 1.0)
    np.testing.assert_allclose(predict_boosting(model, boundary), [0.1 - 256 * 0.03 * 0.02])
    with pytest.raises(FileExistsError):
        save_boosting(path, ridge, [tree()] * 256)
    with pytest.raises(ValueError):
        load_boosting(path, prefix=True)
    for bad in [np.ones((1, 9)), np.full((1, 10), np.nan), np.full((1, 10), 1e300)]:
        with pytest.raises(ValueError):
            predict_boosting(model, bad)


def test_prefix_and_invalid_topology_parameters_archive(tmp_path):
    path = tmp_path / "prefix.npz"
    save_boosting(path, np.zeros(11), [tree()] * 2)
    assert len(load_boosting(path, prefix=True).trees) == 2
    for count in [0, 1, 3, 257]:
        with pytest.raises(ValueError):
            save_boosting(tmp_path / "bad.npz", np.zeros(11), [tree()] * count)
    bad = deepcopy(tree())
    bad["children_left"][0] = 0
    with pytest.raises(ValueError):
        save_boosting(tmp_path / "bad.npz", np.zeros(11), [bad] * 2)
    with pytest.raises(ValueError):
        save_boosting(tmp_path / "bad.npz", np.zeros(10), [tree()] * 2)
    with pytest.raises(ValueError):
        save_boosting(tmp_path / "bad.npz", np.full(11, np.inf), [tree()] * 2)
    deep = {
        "children_left": np.array([1, 3, -1, 5, -1, -1, -1]),
        "children_right": np.array([2, 4, -1, 6, -1, -1, -1]),
        "feature": np.array([0, 0, -2, 0, -2, -2, -2]),
        "threshold": np.zeros(7),
        "value": np.zeros(7),
    }
    with pytest.raises(ValueError, match="depth-two"):
        save_boosting(tmp_path / "deep.npz", np.zeros(11), [deep] * 2)
    np.savez(tmp_path / "extra.npz", surprise=np.zeros(1))
    with pytest.raises(ValueError):
        load_boosting(tmp_path / "extra.npz")


def test_fixed_protocol_constructor_never_fits_and_reuses_original_gates():
    from rl_stocks.training.supervised.boosting_spec import make_boosting, protocol
    from rl_stocks.training.supervised.spec import protocol as original

    spec = protocol()
    assert spec["feasibility_gate"] == original()["feasibility_gate"]
    assert spec["folds"] == original()["folds"] and spec["target"] == original()["target"]
    assert spec["execution"]["allocation_admitted"] is False
    model = make_boosting(23)
    assert model.n_estimators == 256 and model.max_depth == 2 and model.init == "zero"
    assert not hasattr(model, "estimators_")
    assert make_boosting(11, prefix=True).n_estimators == 2
    for seed in [True, 0, 99]:
        with pytest.raises(ValueError):
            make_boosting(seed)
    with pytest.raises(ValueError):
        make_boosting(23, prefix=True)

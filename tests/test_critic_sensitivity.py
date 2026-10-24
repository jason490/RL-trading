import numpy as np
import pytest

from rl_stocks.evaluation.critic_sensitivity import action_change, normal_kl, tensor_change


def test_tensor_norms_and_zero_reference():
    result = tensor_change({"a": [3.0, 4.0]}, {"a": [6.0, 8.0]})
    assert result == dict(
        parameters=2, l2_delta=5.0, maximum_absolute_delta=4.0, relative_l2_delta=1.0
    )
    assert tensor_change({"a": [0.0]}, {"a": [1.0]})["relative_l2_delta"] is None
    with pytest.raises(ValueError):
        tensor_change({"a": [0.0]}, {"b": [0.0]})
    with pytest.raises(ValueError):
        tensor_change({"a": [0.0]}, {"a": [[0.0]]})
    with pytest.raises(ValueError):
        tensor_change({"a": [float("nan")]}, {"a": [0.0]})


def test_action_geometry_and_sign_changes():
    before = np.array([[-0.5, 0, 0, 0], [0.5, 0, 0, 0]])
    after = np.array([[0.5, 0.2, 0, 0], [0.5, 0, 0, 0]])
    result = action_change(before, after)
    assert result["exposure_sign_change_fraction"] == 0.5
    assert result["mean_absolute_delta"] == [0.5, 0.1, 0, 0]
    assert result["maximum_absolute_delta"] == [1.0, 0.2, 0, 0]
    for invalid in (np.ones((2, 3)), np.ones((0, 4)), np.full((2, 4), 2), np.full((2, 4), np.nan)):
        with pytest.raises(ValueError):
            action_change(invalid, invalid)


def test_gaussian_kl_known_shift_and_asymmetry():
    zero = np.zeros((2, 4))
    shifted = np.ones((2, 4))
    assert normal_kl(zero, zero, zero, zero)["mean"] == 0
    assert normal_kl(zero, zero, shifted, zero)["mean"] == 2
    wide = np.full((2, 4), np.log(2))
    assert normal_kl(zero, zero, zero, wide)["mean"] == pytest.approx(4 * (np.log(2) - 0.375))
    assert normal_kl(zero, wide, zero, zero)["mean"] != normal_kl(zero, zero, zero, wide)["mean"]
    with pytest.raises(ValueError):
        normal_kl(zero, zero[:, :3], zero, zero)

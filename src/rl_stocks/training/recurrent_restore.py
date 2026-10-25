"""Shared read-only recurrent checkpoint and frozen sequence comparisons."""

import numpy as np
from rl_stocks.core.contracts import canonical
from rl_stocks.risk.exposure import filter_exposure


def same_restored(actual, recorded):
    """Loading SB3 restores learned tensors/moments, not global sampling streams."""
    return canonical({k: v for k, v in actual.items() if k != "rng"}) == canonical(
        {k: v for k, v in recorded.items() if k != "rng"}
    )


def check_prediction(selected, frozen, observation, expected_state, actual_state, episode_start):
    previous = None if actual_state is None else tuple(s.copy() for s in actual_state)
    expected, left = selected.predict(
        observation,
        state=expected_state,
        episode_start=np.array([episode_start]),
        deterministic=True,
    )
    expected = filter_exposure(expected, frozen.exposure_execution)
    actual, right = frozen.predict_recurrent(
        observation, state=actual_state, episode_start=episode_start
    )
    repeat, again = frozen.predict_recurrent(
        observation, state=actual_state, episode_start=episode_start
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, repeat)
    for a, b, c in zip(left, right, again, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(b, c)
    if previous is not None:
        for old, current in zip(previous, actual_state, strict=True):
            np.testing.assert_array_equal(old, current)
    return actual, left, right

"""Descriptive frozen-policy differences on identical observations; no fitting."""

import numpy as np


def tensor_change(before, after):
    if not before or set(before) != set(after):
        raise ValueError("matching nonempty tensor names required")
    squared = base = 0.0
    maximum = 0.0
    count = 0
    for name in before:
        left = np.asarray(before[name], dtype=np.float64)
        right = np.asarray(after[name], dtype=np.float64)
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("matching finite tensor shapes required")
        difference = right - left
        squared += float(np.square(difference).sum())
        base += float(np.square(left).sum())
        maximum = max(maximum, float(np.abs(difference).max()))
        count += left.size
    return dict(
        parameters=count,
        l2_delta=float(np.sqrt(squared)),
        maximum_absolute_delta=maximum,
        relative_l2_delta=None if base == 0 else float(np.sqrt(squared / base)),
    )


def action_change(before, after):
    left, right = (np.asarray(v, dtype=np.float64) for v in (before, after))
    if (
        left.ndim != 2
        or left.shape[0] == 0
        or left.shape[1] != 4
        or right.shape != left.shape
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or (np.abs(left) > 1).any()
        or (np.abs(right) > 1).any()
    ):
        raise ValueError("matching finite bounded four-component action batches required")
    delta = np.abs(right - left)
    return dict(
        rows=len(left),
        mean_absolute_delta=delta.mean(axis=0).tolist(),
        p95_absolute_delta=np.quantile(delta, 0.95, axis=0).tolist(),
        maximum_absolute_delta=delta.max(axis=0).tolist(),
        exposure_sign_change_fraction=float(np.mean(np.sign(left[:, 0]) != np.sign(right[:, 0]))),
    )


def normal_kl(before_mean, before_log_std, after_mean, after_log_std):
    """Exact KL(before||after) for diagonal pre-tanh Gaussians, summed over actions."""
    values = [
        np.asarray(v, dtype=np.float64)
        for v in (before_mean, before_log_std, after_mean, after_log_std)
    ]
    if (
        values[0].ndim != 2
        or values[0].shape[0] == 0
        or values[0].shape[1] != 4
        or any(v.shape != values[0].shape or not np.isfinite(v).all() for v in values)
    ):
        raise ValueError("matching finite Gaussian parameter batches required")
    m0, s0, m1, s1 = values
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        kl = (
            s1 - s0 + 0.5 * (np.exp(2 * (s0 - s1)) + np.square(m0 - m1) * np.exp(-2 * s1) - 1)
        ).sum(axis=1)
    if not np.isfinite(kl).all() or (kl < -1e-12).any():
        raise ValueError("finite nonnegative Gaussian KL required")
    kl = np.maximum(kl, 0)
    return dict(
        direction="pre_update_to_post_update",
        mean=float(kl.mean()),
        median=float(np.median(kl)),
        maximum=float(kl.max()),
    )

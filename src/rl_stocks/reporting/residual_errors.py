"""Exact saved-forecast correction identities; no fitting or calibration."""

import numpy as np


def residual_effect(target, baseline, prediction, sessions):
    y, b, p = (np.asarray(v, dtype=np.float64) for v in (target, baseline, prediction))
    days = np.asarray(sessions)
    if (
        y.ndim != 1
        or not len(y)
        or b.shape != y.shape
        or p.shape != y.shape
        or days.shape != y.shape
        or any(not np.isfinite(v).all() for v in (y, b, p))
        or any(not isinstance(d, (str, np.str_)) or not d for d in days)
    ):
        raise ValueError("matched finite forecast vectors and session strings required")
    with np.errstate(over="ignore", invalid="ignore"):
        e = y - b
        correction = p - b
        second = float(np.mean(correction**2))
        cross = float(np.mean(e * correction))
        delta = float(np.mean((y - p) ** 2) - np.mean(e**2))
        bias = float(correction.mean() ** 2 - 2 * e.mean() * correction.mean())
        covariance = float(np.mean((e - e.mean()) * (correction - correction.mean())))
        variability = float(correction.var() - 2 * covariance)
        common = within = 0.0
        for day in np.unique(days):
            take = days == day
            weight = float(take.mean())
            ed, cd = e[take], correction[take]
            common += weight * float(cd.mean() ** 2 - 2 * ed.mean() * cd.mean())
            within += weight * float(cd.var() - 2 * np.mean((ed - ed.mean()) * (cd - cd.mean())))
    values = [second, cross, delta, bias, covariance, variability, common, within]
    if not np.isfinite(values).all() or not np.allclose(
        [second - 2 * cross, bias + variability, common + within], delta, rtol=1e-10, atol=1e-13
    ):
        raise ValueError("finite exact residual-MSE identities required")
    return {
        "rows": len(y),
        "sessions": len(np.unique(days)),
        "candidate_minus_baseline_mse": delta,
        "correction_second_moment": second,
        "baseline_error_correction_product_mean": cross,
        "baseline_error_correction_covariance": covariance,
        "pooled_bias_component": bias,
        "pooled_variability_component": variability,
        "daily_common_component": common,
        "daily_within_component": within,
        "interpretation": "Fixed saved forecasts only. Added correction energy minus twice error alignment equals MSE change. No scale/offset is fitted or applied; daily components are descriptive, not independent evidence or trading P&L.",
    }

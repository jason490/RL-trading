"""Descriptive saved-forecast errors; never a fit, correction or trading policy."""
import numpy as np


def decompose(target, prediction, sessions):
    """Return exact pooled and session-level MSE identities, without fitting.

    Negative outcomes are reporting cohorts only. Precision/recall refer to
    gross forecast-target signs and say nothing about executable short profits.
    """
    y=np.asarray(target,dtype=np.float64);p=np.asarray(prediction,dtype=np.float64)
    days=np.asarray(sessions)
    if (y.ndim!=1 or not len(y) or p.shape!=y.shape or days.shape!=y.shape
        or not np.isfinite(y).all() or not np.isfinite(p).all()):
        raise ValueError('matched nonempty finite vectors required')
    if any(not isinstance(d,(str,np.str_)) or not d for d in days):
        raise ValueError('session strings required')
    error=p-y;bias=float(error.mean());vy=float(y.var());vp=float(p.var())
    covariance=float(np.mean((y-y.mean())*(p-p.mean())))
    common=within=0.
    for day in np.unique(days):
        e=error[days==day];weight=len(e)/len(y)
        common+=weight*float(e.mean()**2);within+=weight*float(e.var())
    mse=float(np.mean(error**2))
    if not np.allclose([vy+vp-2*covariance+bias*bias,common+within],mse,rtol=1e-12,atol=1e-14):
        raise ValueError('MSE decomposition failed')
    negative=y<0;forecast_negative=p<0;joint=negative&forecast_negative
    return dict(rows=len(y),sessions=len(np.unique(days)),mean_target=float(y.mean()),
        mean_prediction=float(p.mean()),bias=bias,target_variance=vy,prediction_variance=vp,
        prediction_target_covariance=covariance,mse=mse,
        daily_common_error_mse=common,daily_within_error_mse=within,
        negative_target_rows=int(negative.sum()),negative_forecast_rows=int(forecast_negative.sum()),
        negative_sign_precision=float(joint.sum()/forecast_negative.sum()) if forecast_negative.any() else None,
        negative_sign_recall=float(joint.sum()/negative.sum()) if negative.any() else None)

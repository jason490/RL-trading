"""Descriptive numeric SAC target decomposition; never an optimizer."""
import math
import numpy as np


def moments(values):
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError('nonempty finite vector required')
    return dict(count=len(a),mean=float(a.mean()),mean_abs=float(np.abs(a).mean()),std=float(a.std()),
        minimum=float(a.min()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(a.max()))


def target_parts(reward,next_q,next_log_prob,dones,timeouts,alpha,gamma):
    arrays=[np.asarray(v,dtype=np.float64) for v in (reward,next_q,next_log_prob,dones,timeouts)]
    r,q,logp,done,timeout=arrays
    if (r.ndim!=1 or not len(r) or any(a.shape!=r.shape or not np.isfinite(a).all() for a in arrays)
        or not math.isfinite(alpha) or alpha<=0 or not math.isfinite(gamma) or not 0<=gamma<1
        or not np.isin(done,[0,1]).all() or not np.isin(timeout,[0,1]).all() or (timeout>done).any()):
        raise ValueError('matched finite one-step SAC inputs required')
    continuation=1-done*(1-timeout)
    boot=gamma*continuation*q;entropy=-gamma*continuation*alpha*logp
    return dict(reward=r,discounted_target_q=boot,discounted_entropy=entropy,target=r+boot+entropy)

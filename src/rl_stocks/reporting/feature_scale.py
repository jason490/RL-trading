"""Outcome-free frozen-normalizer distribution summaries; never fit preprocessing."""
import numpy as np
from rl_stocks.data.generation.datasets import normalizer_from


def summarize_scale(rows,names,normalizer):
    frozen=normalizer_from(normalizer)
    if tuple(names)!=frozen.names:raise ValueError('ordered feature schema required')
    values=np.asarray(rows,dtype=np.float64)
    if values.ndim!=2 or values.shape[0]==0 or values.shape[1]!=len(names) or not np.isfinite(values).all():
        raise ValueError('nonempty finite feature matrix required')
    z=(values-np.asarray(frozen.means))/np.asarray(frozen.scales)
    if getattr(frozen,'clip_abs',None) is not None:
        z=np.clip(z,-frozen.clip_abs,frozen.clip_abs)
    if not np.isfinite(z).all():raise ValueError('finite normalized values required')
    result={}
    for i,name in enumerate(names):
        column=z[:,i];quantiles=np.quantile(column,[.01,.5,.99],method='linear')
        row=dict(count=len(column),mean=float(column.mean()),std=float(column.std()),
            minimum=float(column.min()),maximum=float(column.max()),maximum_abs=float(np.abs(column).max()),
            p01=float(quantiles[0]),median=float(quantiles[1]),p99=float(quantiles[2]),
            **{f'fraction_abs_above_{cutoff}':float(np.mean(np.abs(column)>cutoff)) for cutoff in (3,5,10)})
        if not all(np.isfinite(v) for v in row.values()):raise ValueError('finite scale summary required')
        result[name]=row
    return result

"""Fixed same-session forecast transformation, with no label or fitting interface."""
import numpy as np
from rl_stocks.data.validation.bars import timestamp


def center_forecasts(prediction,ridge,sessions,symbols,available,entry_sessions,expected_symbols):
    raw=np.asarray(prediction,dtype=np.float64);anchor=np.asarray(ridge,dtype=np.float64)
    arrays=[np.asarray(a) for a in (sessions,symbols,available,entry_sessions)]
    if (raw.ndim!=1 or anchor.shape!=raw.shape or not len(raw)
        or any(a.shape!=raw.shape for a in arrays) or not np.isfinite(raw).all() or not np.isfinite(anchor).all()):
        raise ValueError('finite aligned forecast panel required')
    sessions,symbols,available,entry_sessions=arrays;expected=set(expected_symbols)
    if not expected or len(expected)!=len(expected_symbols):raise ValueError('unique expected universe required')
    out=np.empty_like(raw);proof=[]
    for day in sorted(set(sessions)):
        take=np.flatnonzero(sessions==day)
        if len(take)!=len(expected) or set(symbols[take])!=expected:raise ValueError('complete unique same-session universe required')
        times=[timestamp(str(v)) for v in available[take]];cutoff=max(times)
        if any(str(v)[:10]!=str(day) for v in available[take]):raise ValueError('feature availability outside decision session')
        if any(cutoff>=timestamp(str(d)+'T00:00:00+00:00') for d in entry_sessions[take]):
            raise ValueError('panel not available before next session')
        raw_mean=float(raw[take].mean());ridge_mean=float(anchor[take].mean())
        out[take]=raw[take]-raw_mean+ridge_mean
        if not np.isclose(out[take].mean(),ridge_mean,rtol=0,atol=1e-12):raise ValueError('centering mean drift')
        proof.append({'session':str(day),'decision_cutoff':cutoff.isoformat(),'symbols':len(take)})
    return out,proof

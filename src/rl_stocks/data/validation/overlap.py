"""Check independently retrieved price units without modifying either history."""
import math
from rl_stocks.data.validation.bars import validate


def compare_overlap(reference, candidate, *, price_atol=1e-6,price_rtol=1e-7,dividend_atol=1e-8):
    validate(reference);validate(candidate)
    if any(not math.isfinite(v) or v<0 for v in (price_atol,price_rtol,dividend_atol)):
        raise ValueError('finite nonnegative tolerances required')
    left={(b.symbol,b.session):b for b in reference};right={(b.symbol,b.session):b for b in candidate}
    if set(left)!=set(right):raise ValueError('overlap row sets differ')
    mismatches=[];count=0;max_abs={k:0. for k in ('open','high','low','close','volume','dividend')}
    for key,a in left.items():
        b=right[key]
        for field in ('provider','feed','adjustments','open_time','event_time','available_time','split'):
            if getattr(a,field)!=getattr(b,field):
                count+=1
                if len(mismatches)<20:mismatches.append(dict(symbol=key[0],session=key[1],field=field,reference=getattr(a,field),candidate=getattr(b,field)))
        for field in max_abs:
            av=getattr(a,field);bv=getattr(b,field);delta=abs(av-bv);max_abs[field]=max(max_abs[field],delta)
            tolerance=price_atol+price_rtol*abs(av) if field in ('open','high','low','close') else dividend_atol if field=='dividend' else 0.
            if delta>tolerance:
                count+=1
                if len(mismatches)<20:mismatches.append(dict(symbol=key[0],session=key[1],field=field,reference=av,candidate=bv))
    return dict(passed=count==0,rows=len(left),mismatch_count=count,mismatches_first20=mismatches,
        maximum_absolute_difference=max_abs,tolerances=dict(price_atol=price_atol,price_rtol=price_rtol,dividend_atol=dividend_atol),
        rescaling_applied=False,reference_modified=False,candidate_modified=False)

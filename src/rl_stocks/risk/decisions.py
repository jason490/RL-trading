from dataclasses import asdict, dataclass
import math
from rl_stocks.core.contracts import ACTION_SCHEMA

class GeometryError(ValueError):
    """Finite policy proposal cannot produce positive protective prices."""

@dataclass(frozen=True)
class Decision:
    schema: str
    symbol: str
    decision_time: str
    expiry_time: str
    reference: float
    raw: tuple[float,...]
    approved: tuple[float,...]
    exposure: float
    quantity: int
    stop: float | None
    near: float | None
    far: float | None
    direction: int
    intent: str
    clipped: bool
    notional: float
    zone_lower: float | None
    zone_upper: float | None
    side: str

def decode(action, symbol, decision_time, expiry_time, reference, atr, equity, current_quantity=0):
    raw=tuple(float(x) for x in action)
    if len(raw)!=4 or not all(math.isfinite(x) for x in raw):
        raise ValueError("action must contain four finite values")
    if not all(math.isfinite(x) and x>0 for x in (reference,atr,equity)):
        raise ValueError("invalid reference/ATR/equity")
    a=tuple(max(-1.,min(1.,x)) for x in raw)
    direction=1 if a[0]>0 else -1 if a[0]<0 else 0
    quantity=direction*int(abs(a[0])*equity/reference)
    if quantity==0:direction=0
    # Policy learns each independent distance; no fixed reward-to-risk formula.
    stop_distance=(.5+(a[1]+1)/2*4.5)*atr
    near_distance=(.5+(a[2]+1)/2*7.5)*atr
    width=(.25+(a[3]+1)/2*3.75)*atr
    stop=reference-direction*stop_distance if direction else None
    near=reference+direction*near_distance if direction else None
    far=reference+direction*(near_distance+width) if direction else None
    if direction and min(stop,near,far)<=0:
        raise GeometryError("nonpositive protective price")
    intent=("hold" if quantity==current_quantity else "close" if quantity==0 else
            "reverse" if quantity*current_quantity<0 else "add" if abs(quantity)>abs(current_quantity) else "reduce")
    return Decision(ACTION_SCHEMA,symbol,decision_time,expiry_time,reference,raw,a,a[0],
                    quantity,stop,near,far,direction,intent,raw!=a,abs(quantity)*reference,
                    min(near,far) if direction else None,max(near,far) if direction else None,
                    "long" if direction>0 else "short" if direction<0 else "flat")

def valid_entry(decision, price):
    if decision.direction==0:
        return True
    return (decision.stop<price<decision.near<decision.far if decision.direction>0
            else decision.far<decision.near<price<decision.stop)

from dataclasses import asdict, dataclass, replace
import math
from rl_stocks.environment.accounting.portfolio import Portfolio
from rl_stocks.risk.decisions import decode, valid_entry, GeometryError
from rl_stocks.data.validation.bars import timestamp

@dataclass(frozen=True)
class SimulationConfig:
    initial_equity: float=100000.
    fee_bps: float=1.
    impact_bps: float=2.  # half spread plus slippage, adverse on every fill
    borrow_rate: float=.03
    max_holding: int=20
    allow_short: bool=True  # simulation assumption, not broker eligibility
    exposure_cap: float=1.
    def __post_init__(self):
        if not math.isfinite(self.initial_equity) or self.initial_equity<=0 or not 1<=self.max_holding<=20 or not 0<self.exposure_cap<=1:
            raise ValueError("invalid simulation limits")
        if not all(math.isfinite(x) and x>=0 for x in (self.fee_bps,self.impact_bps,self.borrow_rate)):
            raise ValueError("negative costs")

class Simulator:
    def __init__(self,config=SimulationConfig()):
        self.config=config
        self.portfolio=Portfolio(cash=config.initial_equity)
        self.protection=None
        self.near_done=False
        self.last_protective_action=None
    def trade(self,target,price,time,reason):
        self.portfolio.trade(target,price,time,reason,self.config.fee_bps,self.config.impact_bps)
        if target==0:
            self.protection=None;self.near_done=False;self.last_protective_action=None
    def protect(self,bar,gap_only=False):
        p=self.portfolio
        if not p.quantity or self.protection is None:return False
        stop,near,far=self.protection
        long=p.quantity>0
        stop_hit=(bar.open<=stop if long else bar.open>=stop) if gap_only else (bar.low<=stop if long else bar.high>=stop)
        if stop_hit:
            price=min(bar.open,stop) if long else max(bar.open,stop)
            self.trade(0,price,bar.open_time if gap_only else bar.event_time,"stop")
            return True
        hit=lambda level: (bar.open>=level if long else bar.open<=level) if gap_only else (bar.high>=level if long else bar.low<=level)
        near_hit=False
        if not self.near_done and hit(near):
            close=max(1,abs(p.quantity)//2)
            self.trade(p.quantity-(close if long else -close),near,bar.event_time,"target_near")
            self.near_done=True
            near_hit=True
        if p.quantity and hit(far):
            self.trade(0,far,bar.event_time,"target_far")
            return True
        return near_hit
    def advance(self,action,previous,bar,atr):
        p=self.portfolio;c=self.config
        old_equity=p.equity(previous.close)
        invalid_geometry=False
        try:
            decision=decode(action,previous.symbol,previous.available_time,bar.event_time,
                            previous.close,atr,max(old_equity,1e-8),p.quantity)
        except GeometryError:
            decision=None;invalid_geometry=True
        if timestamp(previous.available_time)>=timestamp(bar.open_time):
            raise ValueError("bar not available before execution open")
        p.corporate_action(bar)
        if bar.split!=1 and decision is not None:
            decision=replace(decision,reference=decision.reference/bar.split,
                quantity=math.trunc(decision.quantity*bar.split),
                stop=decision.stop/bar.split if decision.stop is not None else None,
                near=decision.near/bar.split if decision.near is not None else None,
                far=decision.far/bar.split if decision.far is not None else None,
                zone_lower=decision.zone_lower/bar.split if decision.zone_lower is not None else None,
                zone_upper=decision.zone_upper/bar.split if decision.zone_upper is not None else None)
        if bar.split!=1 and self.protection:
            self.protection=tuple(x/bar.split for x in self.protection)
        # Existing protections remain active through overnight gaps, before adjustments.
        exited=self.protect(bar,gap_only=True)
        rejected="nonpositive_protective_price" if invalid_geometry else "protective_fill_precedes_signal" if exited else None
        approved_target=None
        if not exited and decision is not None:
            if decision.direction<0 and not c.allow_short:
                rejected="short_disabled"
            elif not valid_entry(decision,bar.open):
                rejected="gap_invalid_geometry"
            else:
                target=decision.quantity
                # Decision quantity uses yesterday's known equity/price; the execution
                # gate can only reduce it for today's gap and actual buying power.
                affordable=max(0,int(p.equity(bar.open)*c.exposure_cap/
                    (bar.open*(1+(c.fee_bps+c.impact_bps)/10000))))
                target=max(-affordable,min(affordable,target))
                approved_target=target
                if target*p.quantity<0:
                    self.trade(0,bar.open,bar.open_time,"reverse_close")
                opened=p.quantity==0 and target!=0
                self.trade(target,bar.open,bar.open_time,decision.intent)
                if p.quantity and (opened or self.last_protective_action!=decision.approved[1:]):
                    self.protection=(decision.stop,decision.near,decision.far)
                    self.near_done=False
                    self.last_protective_action=decision.approved[1:]
        self.protect(bar)
        if p.quantity:
            p.age+=1
            if p.age>=c.max_holding:
                self.trade(0,bar.close,bar.event_time,"holding_limit")
        borrowed=p.borrow(bar.close,c.borrow_rate)
        equity=p.equity(bar.close)
        if not math.isfinite(equity):raise ValueError("nonfinite equity")
        return {"equity":equity,"cash":p.cash,"quantity":p.quantity,"age":p.age,
            "exposure":p.quantity*bar.close/max(equity,1e-8),"costs":p.costs,
            "turnover":p.turnover,"borrow_cost":borrowed,"rejected":rejected,
            "decision":asdict(decision) if decision is not None else None,
            "raw_action":tuple(float(x) for x in action),"protection":self.protection,
            "approved_target_quantity":approved_target,
            "risk_clipped":approved_target is not None and approved_target!=decision.quantity,
            "raw_return":equity/old_equity-1 if old_equity>0 else -1,
            "previous_equity":old_equity}

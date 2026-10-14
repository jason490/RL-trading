from dataclasses import dataclass, field
import math

@dataclass
class Portfolio:
    cash: float = 100000.
    quantity: int = 0
    basis: float = 0.
    age: int = 0
    costs: float = 0.
    turnover: float = 0.
    events: list = field(default_factory=list)
    def equity(self,price):
        return self.cash+self.quantity*price
    def trade(self,target,price,time,reason,fee_bps=1.,impact_bps=2.):
        delta=target-self.quantity
        if delta==0:return
        sign=1 if delta>0 else -1
        fill=price*(1+sign*impact_bps/10000)
        fee=abs(delta)*fill*fee_bps/10000
        old=self.quantity
        closed=min(abs(old),abs(delta)) if old*delta<0 else 0
        pnl=closed*(fill-self.basis)*(1 if old>0 else -1) if closed else 0.
        old_age=self.age
        if old==0 or old*target<0:
            self.basis=fill;self.age=0
        elif abs(target)>abs(old):
            self.basis=(abs(old)*self.basis+(abs(target)-abs(old))*fill)/abs(target)
        elif target==0:
            self.basis=0.;self.age=0
        self.cash-=delta*fill+fee
        self.quantity=target
        self.turnover+=abs(delta)*price
        impact=abs(delta)*abs(fill-price)
        self.costs+=fee+impact
        self.events.append(dict(time=time,reason=reason,delta=delta,quantity=target,price=fill,
            reference_price=price,fee=fee,impact=impact,closed=closed,realized_pnl=pnl-fee,
            holding_sessions=old_age if closed else None,equity=self.equity(price)))
    def corporate_action(self,bar):
        if bar.split!=1:
            exact=self.quantity*bar.split
            shares=math.trunc(exact)
            self.cash+=(exact-shares)*bar.open
            self.quantity=shares
            self.basis/=bar.split
        self.cash+=self.quantity*bar.dividend
    def borrow(self,price,rate=.03):
        cost=abs(min(0,self.quantity))*price*rate/252
        self.cash-=cost;self.costs+=cost
        return cost

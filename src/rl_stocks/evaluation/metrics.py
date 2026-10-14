import math
from statistics import mean, stdev

def metrics(rows,events,initial=100000.):
    equity=[initial]+[r["equity"] for r in rows]
    returns=[b/a-1 for a,b in zip(equity,equity[1:]) if a>0]
    peak=initial;drawdown=[]
    for value in equity:
        peak=max(peak,value);drawdown.append(value/peak-1)
    sigma=stdev(returns) if len(returns)>1 else 0
    exits=[e for e in events if e.get("closed",0)>0]
    profits=[e["realized_pnl"] for e in exits]
    wins=[p for p in profits if p>0];losses=[p for p in profits if p<0]
    return dict(net_return=equity[-1]/initial-1,max_drawdown=min(drawdown),
        volatility=sigma*math.sqrt(252),sharpe=mean(returns)/sigma*math.sqrt(252) if sigma else 0.,
        annualization_sessions=252,risk_free_rate=0.,turnover=rows[-1].get("turnover",0)/initial if rows else 0.,
        costs=rows[-1].get("costs",0) if rows else 0.,exit_fills=len(profits),
        win_rate=len(wins)/len(profits) if profits else 0.,
        average_win=mean(wins) if wins else 0.,average_loss=mean(losses) if losses else 0.,
        profit_factor=sum(wins)/abs(sum(losses)) if losses else None,
        expectancy=mean(profits) if profits else 0.,
        trade_diagnostics="Exit-fill diagnostics; partial exits counted separately; fees on entry excluded from realized diagnostics, included in equity")

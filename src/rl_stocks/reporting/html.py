from pathlib import Path
import json
import html

def report(results,destination,title,checkpoints=None):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        from rl_stocks.reporting.offline import report as offline_report
        return offline_report(results,destination,title,checkpoints)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    sections=["<!doctype html><meta charset='utf-8'><title>"+html.escape(title)+"</title>",
        "<h1>"+html.escape(title)+"</h1><p>Research simulation. Costs included. 252 sessions/year. No profitability claim.</p>"]
    first=True
    for key,r in results.items():
        rows=r["rows"];events=r["events"]
        fig=make_subplots(rows=4,cols=1,shared_xaxes=True,
            subplot_titles=("Equity and buy-and-hold","Drawdown","Price, stops, learned target zone and fills","Exposure"))
        dates=[x["session"] for x in rows];equity=[x["equity"] for x in rows]
        fig.add_trace(go.Scatter(x=dates,y=equity,name="Policy equity"),row=1,col=1)
        base=r["baselines"]["buy_hold"]["rows"]
        fig.add_trace(go.Scatter(x=[x["session"] for x in base],y=[x["equity"] for x in base],name="Buy hold"),row=1,col=1)
        peak=100000.;dd=[]
        for e in equity:peak=max(peak,e);dd.append(e/peak-1)
        fig.add_trace(go.Scatter(x=dates,y=dd,name="Drawdown",fill="tozeroy"),row=2,col=1)
        fig.add_trace(go.Scatter(x=dates,y=[x["price"] for x in rows],name="Close"),row=3,col=1)
        for i,label in enumerate(("Stop","Near","Far")):
            fig.add_trace(go.Scatter(x=dates,y=[x["protection"][i] if x["protection"] else None for x in rows],
                name=label,connectgaps=False,fill="tonexty" if label=="Far" else None),row=3,col=1)
        fig.add_trace(go.Scatter(x=[e["time"][:10] for e in events],y=[e["price"] for e in events],mode="markers",
            text=[e["reason"] for e in events],name="Fills"),row=3,col=1)
        fig.add_trace(go.Scatter(x=dates,y=[x["exposure"] for x in rows],name="Exposure"),row=4,col=1)
        fig.update_layout(height=1100,title=f"{key}: {r['split']}")
        sections.append(fig.to_html(full_html=False,include_plotlyjs=True if first else False));first=False
        diag=make_subplots(rows=1,cols=2,subplot_titles=("Holding sessions at exits","Exit-fill P&L"))
        diag.add_trace(go.Histogram(x=[e["holding_sessions"] for e in events if e.get("closed")]),row=1,col=1)
        diag.add_trace(go.Histogram(x=[e["realized_pnl"] for e in events if e.get("closed")]),row=1,col=2)
        sections.append(diag.to_html(full_html=False,include_plotlyjs=False))
        sections.append("<pre>"+html.escape(json.dumps({"policy":r["metrics"],
            "baselines":{k:v["metrics"] for k,v in r["baselines"].items()},
            "rejected_decisions":sum(x["rejected"] is not None for x in rows)},indent=2))+"</pre>")
    if checkpoints:
        fig=go.Figure(go.Scatter(x=[x["steps"] for x in checkpoints],y=[x["score"] for x in checkpoints],mode="lines+markers"))
        fig.update_layout(title="Validation checkpoint selection: mean net return")
        sections.append(fig.to_html(full_html=False,include_plotlyjs=False))
    destination.write_text("\n".join(sections))
    destination.with_suffix(".json").write_text(json.dumps(results,indent=2,allow_nan=False))

"""Self-contained interactive SVG fallback when Plotly is unavailable."""
from collections import Counter
import html
import json
import math
from pathlib import Path

COLORS = ('#2563eb','#059669','#dc2626','#a855f7','#d97706')


def chart(title, dates, series):
    valid = [v for _,values in series for v in values if v is not None and math.isfinite(v)]
    if not valid:
        return '<p>No values for '+html.escape(title)+'</p>'
    lo,hi = min(valid),max(valid)
    pad = max((hi-lo)*.05,abs(hi)*.001,1e-6)
    lo-=pad;hi+=pad
    def x(i): return 60+i*880/max(1,len(dates)-1)
    def y(v): return 240-(v-lo)/(hi-lo)*210
    result = ["<figure><figcaption>"+html.escape(title)+"</figcaption>",
              "<svg viewBox='0 0 1000 280' role='img' aria-label='"+html.escape(title,quote=True)+"'>"]
    for n in range(5):
        value=lo+(hi-lo)*n/4
        result.append(f"<line x1='60' x2='940' y1='{y(value):.2f}' y2='{y(value):.2f}' stroke='#ddd'/><text x='2' y='{y(value):.2f}'>{value:.4g}</text>")
    for label,values in series:
        color=COLORS[series.index((label,values))%len(COLORS)]
        points=[]
        for i,value in enumerate(values):
            if value is None:
                if points:
                    result.append(f"<polyline class='series' points='{' '.join(points)}' fill='none' stroke='{color}' stroke-width='2'/>")
                    points=[]
                continue
            points.append(f'{x(i):.2f},{y(value):.2f}')
            result.append(f"<circle class='series' cx='{x(i):.2f}' cy='{y(value):.2f}' r='3' fill='{color}'><title>{html.escape(label)} {html.escape(dates[i])}: {value:.6g}</title></circle>")
        if points:result.append(f"<polyline class='series' points='{' '.join(points)}' fill='none' stroke='{color}' stroke-width='2'/>")
        result.append(f"<text x='60' y='{16+series.index((label,values))*14}' fill='{color}'>{html.escape(label)}</text>")
    for i in sorted({0,len(dates)//2,len(dates)-1}):
        result.append(f"<text x='{x(i):.2f}' y='270' text-anchor='middle'>{html.escape(dates[i])}</text>")
    result.append('</svg></figure>')
    return ''.join(result)


def report(results,destination,title,checkpoints=None):
    sections=["<!doctype html><html lang='en'><meta charset='utf-8'><title>"+html.escape(title)+"</title>",
        "<style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:20px;background:#fafafa}svg{width:100%;background:white}svg text{font-size:12px}figure{margin:24px 0}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}pre{white-space:pre-wrap}circle{opacity:.15}circle:hover{opacity:1}</style>",
        '<h1>'+html.escape(title)+'</h1><p>Research simulation; no profitability evidence. Native SVG fallback because Plotly is unavailable. Hover points for values; expand tables for fills and decisions.</p>']
    for key,result in results.items():
        rows=result['rows'];dates=[r['session'] for r in rows]
        sections.append('<h2>'+html.escape(key)+' ('+html.escape(result['split'])+')</h2>')
        base=result['baselines']['buy_hold']['rows']
        baseline_by_session={r['session']:r['equity'] for r in base}
        sections.append(chart('Equity',dates,[('Policy',[r['equity'] for r in rows]),
            ('Buy hold',[baseline_by_session.get(day) for day in dates])]))
        peak=100000;dd=[]
        for r in rows:peak=max(peak,r['equity']);dd.append(r['equity']/peak-1)
        sections.append(chart('Drawdown',dates,[('Drawdown',dd)]))
        series=[('Close',[r['price'] for r in rows])]
        for i,label in enumerate(('Stop','Target near','Target far')):
            series.append((label,[r['protection'][i] if r['protection'] else None for r in rows]))
        sections.append(chart('Price and learned protection',dates,series))
        sections.append(chart('Exposure',dates,[('Exposure',[r['exposure'] for r in rows])]))
        for field,title_ in [('holding_sessions','Holding sessions at exits'),('realized_pnl','Exit-fill P&L')]:
            values=[e[field] for e in result['events'] if e.get('closed')]
            counts=Counter(values if field=='holding_sessions' else [round(v,-2) for v in values])
            labels=sorted(counts)
            sections.append(chart(title_,[str(v) for v in labels],[('Count',[counts[v] for v in labels])]) if labels else '<p>No exits</p>')
        sections.append('<pre>'+html.escape(json.dumps(result['metrics'],indent=2))+'</pre>')
        for label,data in [('Fills',result['events']),('Raw and approved decisions',rows)]:
            sections.append('<details><summary>'+label+'</summary><pre>'+html.escape(json.dumps(data,indent=2))+'</pre></details>')
    if checkpoints:
        sections.append(chart('Validation checkpoint scores',[str(c['steps']) for c in checkpoints],
            [('Mean net return',[c['score'] for c in checkpoints])]))
    sections.append('</html>')
    target=Path(destination);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text('\n'.join(sections))
    target.with_suffix('.json').write_text(json.dumps(results,indent=2,allow_nan=False))

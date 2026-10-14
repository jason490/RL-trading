"""Validation-only run comparisons, including failures and unmatched groups."""
from collections import defaultdict
import html
import json
from pathlib import Path
from statistics import mean, stdev
from rl_stocks.core.contracts import digest
from rl_stocks.reporting.offline import chart


def compare(runs_root, destination):
    rows=[]
    for path in sorted(Path(runs_root).glob('*/metadata.json')):
        metadata=json.loads(path.read_text())
        row={'run':path.parent.name,**metadata}
        row['comparison_group']=digest({'dataset':metadata.get('dataset'),
            'simulation':metadata.get('simulation'),'steps':metadata.get('requested_steps'),
            'smoke':metadata.get('smoke'),'selection_metric':metadata.get('selection_metric')})
        rows.append(row)
    if not rows:
        raise ValueError('no run metadata found')
    groups=defaultdict(list)
    for row in rows:groups[(row['comparison_group'],row['algorithm'])].append(row)
    summaries=[]
    sections=['<!doctype html><meta charset="utf-8"><title>Validation comparisons</title>',
        '<h1>Validation comparisons</h1><p>Matched groups share data, costs, transition budget and selection protocol. Validation selection is not a held-out test result. Failed runs remain visible. Sample standard deviation describes seeds present; a single seed is not a variability estimate.</p>']
    for (group,algorithm),entries in groups.items():
        completed=[r for r in entries if r['status']=='complete']
        scores=[r['selected_score'] for r in completed]
        summary={'group':group,'algorithm':algorithm,'runs':len(entries),'completed':len(completed),
            'failed':sum(r['status']=='failed' for r in entries),'seeds':[r['seed'] for r in completed],
            'mean_validation_score':mean(scores) if scores else None,
            'seed_standard_deviation':stdev(scores) if len(scores)>1 else None}
        summaries.append(summary)
        sections.append('<h2>'+html.escape(algorithm)+' / '+group[:12]+'</h2>')
        sections.append('<pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>')
        if scores:
            sections.append(chart('Validation selection per seed',[str(r['seed']) for r in completed],[('Selected net return',scores)]))
        for row in entries:
            history=row.get('validation_checkpoints',[])
            if history:
                sections.append(chart(row['run'],[str(c['steps']) for c in history],[('Validation score',[c['score'] for c in history])]))
    sections.append('<details open><summary>All runs, including failures</summary><pre>'+html.escape(json.dumps(rows,indent=2))+'</pre></details>')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text('\n'.join(sections))
    destination.with_suffix('.json').write_text(json.dumps({'runs':rows,'summaries':summaries},indent=2,allow_nan=False))
    return destination

"""Validation-only matched family comparison and predeclared candidate selection."""
from collections import defaultdict
import html
import json
from pathlib import Path
from statistics import mean, stdev
from rl_stocks.core.contracts import canonical
from rl_stocks.reporting.offline import chart


def summarize(runs, protocol):
    names = {identity:name for name,identity in protocol["datasets"].items()}
    groups = defaultdict(list)
    seen = set()
    reference = None
    for run in runs:
        if run.get("smoke") or run.get("dataset") not in names:
            continue
        key = (names[run["dataset"]],run["algorithm"],run["seed"])
        if key in seen:
            raise ValueError("duplicate research cell; explicitly select a retry before comparison")
        seen.add(key)
        if run["status"] != "complete" or run["actual_steps"] != protocol["transitions_per_run"]:
            raise ValueError("incomplete research cell")
        assumptions=(run["simulation"],run["selection_metric"],run["source_hash"],run["versions"])
        if reference is None:
            reference=assumptions
        elif assumptions != reference:
            raise ValueError("unmatched costs, selection, source or dependency assumptions")
        groups[key[:2]].append(run)
    expected={(d,a,s) for d in protocol["datasets"] for a in protocol["algorithms"] for s in protocol["seeds"]}
    if seen != expected:
        raise ValueError("matched research matrix is incomplete")
    summaries=[]
    for (dataset,algorithm),entries in sorted(groups.items()):
        entries.sort(key=lambda r:r["seed"])
        scores=[r["selected_score"] for r in entries]
        summaries.append({"dataset":dataset,"algorithm":algorithm,"seeds":[r["seed"] for r in entries],
            "mean_validation_return":mean(scores),"seed_standard_deviation":stdev(scores),
            "scores":scores,"bundle_ids":[r["bundle_id"] for r in entries]})
    winner=sorted(summaries,key=lambda r:(-r["mean_validation_return"],r["dataset"],r["algorithm"]))[0]
    candidates=groups[(winner["dataset"],winner["algorithm"])]
    candidate=sorted(candidates,key=lambda r:(-r["selected_score"],r["seed"]))[0]
    return {"protocol":protocol,"families":summaries,"selected_family":winner,
            "candidate":{"dataset":winner["dataset"],"algorithm":candidate["algorithm"],
                         "seed":candidate["seed"],"bundle_id":candidate["bundle_id"],
                         "validation_score":candidate["selected_score"]},
            "test_evaluated":False}


def report_study(runs, protocol, destination):
    value=summarize(runs,protocol)
    value["runs"]=[{k:r[k] for k in ("algorithm","seed","dataset","actual_steps","selected_step",
        "selected_score","selected_metrics","bundle_id","duration_seconds","git","source_hash")}
        for r in runs if not r.get("smoke") and r.get("dataset") in protocol["datasets"].values()]
    sections=['<!doctype html><meta charset="utf-8"><title>Expanded-universe validation study</title>',
        '<h1>Expanded-universe validation study</h1><p>26 separate symbol episodes. Net modeled costs. '
        'Validation selected checkpoints and families; these are not independent test results. '
        'Retrospective surviving-stock universe. Three seeds per family.</p>']
    families=value["families"]
    sections.append(chart('Mean selected validation return',
        [r["dataset"]+'/'+r["algorithm"] for r in families],
        [('Mean net return',[r["mean_validation_return"] for r in families])]))
    sections.append('<pre>'+html.escape(json.dumps(value,indent=2))+'</pre>')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text('\n'.join(sections))
    destination.with_suffix('.json').write_bytes(canonical(value))
    return value

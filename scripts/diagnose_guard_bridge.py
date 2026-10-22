"""Fixed bar-backed bridge for four accepted update-guard phases; no learning."""
import argparse,json
from pathlib import Path
from statistics import mean
from diagnose_recurrent_actions import audit
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.report_contract import validate_development_report
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.pnl_bridge import COMPONENTS,bridge,holding_bridge


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new output required')
    accepted=audit('guard');state=json.loads(Path('.runtime/guard-screen-v2-state.json').read_text())
    protocol=json.loads(Path('results/recurrent-guard-2026-10-22/summary.json').read_text())
    stocks={v['symbol'] for v in json.loads(Path('configs/universes/sector_research.json').read_text())['instruments'] if v['kind']=='stock'}
    cells=[];datasets={}
    for item in state['completed'].values():
        path=next((Path(item['root'])/'runs').glob('*/metadata.json'));m=json.loads(path.read_text())
        arm=next(k for k,v in protocol['cells'].items() if v['run']==path.parent.name)
        if m['dataset'] not in datasets:datasets[m['dataset']]=read_dataset(Path('data/datasets')/m['dataset'])
        data=datasets[m['dataset']];config=SimulationConfig(**m['simulation'])
        for phase,name in [('selected','selected-validation.json'),('final','validation.json')]:
            results=json.loads((path.parent/name).read_text());validate_development_report(results,data,config);records=[]
            for key,result in results.items():
                episode=data['episodes'][key]
                policy=bridge(episode,result['rows'],result['events'],config.initial_equity)
                holding=holding_bridge(episode,result['baselines']['buy_hold'],config)
                records.append(dict(symbol=result['symbol'],sessions=len(policy['rows']),policy=policy,holding=holding,
                    excess_components_pp={k:policy['components_pp'][k]-holding['components_pp'][k] for k in COMPONENTS}))
            groups={}
            for kind in ['stock','etf']:
                members=[r for r in records if (r['symbol'] in stocks)==(kind=='stock')]
                groups[kind]=dict(count=len(members),excess_components_pp={k:mean(v['excess_components_pp'][k] for v in members) for k in COMPONENTS},
                    max_session_error=max(v[side]['max_session_error'] for v in members for side in ['policy','holding']))
            # Keep compact component totals, no per-session bar-derived records in Git.
            for r in records:
                for side in ['policy','holding']:r[side].pop('rows')
            cells.append(dict(arm=arm,phase=phase,groups=groups,per_symbol=records))
    if any(checksum(Path(p))!=h for p,h in accepted['source_files_sha256'].items()):raise ValueError('source changed during bridge')
    out=dict(records=cells,accepted_summary_sha256=accepted['accepted_summary_sha256'],source_files_sha256=accepted['source_files_sha256'],source_files_unchanged=True,
        source_sha256={str(p):checksum(p) for p in [Path(__file__),Path('src/rl_stocks/reporting/pnl_bridge.py')]},
        training_performed=False,test_evaluated=False,colab_allocated=False,immutable_development_bars_opened=True,objective_achieved=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('Reconciled four accepted phases against immutable development bars; no learning.')


if __name__=='__main__':main()

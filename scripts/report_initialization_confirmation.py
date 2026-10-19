"""Verify and report confirmation of the single predeclared promoted PPO arm."""
import argparse
import json
import math
from pathlib import Path

from rl_stocks.core.contracts import digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.initialization import group_metrics, confirmation_gate
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.training.initialization import initialization_spec
from rl_stocks.training.curves import latest_points


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',type=Path,required=True)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    screen=json.loads(args.screen.read_text());protocol=screen['protocol']
    if digest(protocol)!=screen['protocol_sha256'] or not screen['screen']['passed']:
        raise ValueError('matching passing screen required')
    candidate=screen['screen']['promoted_candidate'];arms=('standard',candidate)
    cells={a:{str(protocol['screen_seed']):screen['cells'][a]} for a in arms}
    state=json.loads(args.state.read_text())
    if state['status']!='complete' or not state['cleanup_verified']:
        raise ValueError('complete confirmation and verified cleanup required')
    universe=json.loads(Path(protocol['universe']).read_text())
    stocks={i['symbol'] for i in universe['instruments'] if i['kind']=='stock'}
    for item in state['completed'].values():
        paths=list((Path(item['root'])/'runs').glob('*/metadata.json'))
        if len(paths)!=1:raise ValueError('one run per confirmation task required')
        path=paths[0];meta=json.loads(path.read_text());arm=meta['initialization']['arm'];seed=str(meta['seed'])
        if (arm not in arms or seed in cells[arm] or meta['seed'] not in (23,37)
                or meta['status']!='complete' or meta['smoke'] or meta['algorithm']!='ppo'
                or meta['actual_steps']!=50000 or meta['requested_steps']!=50000 or meta['device']!='cpu'
                or meta['initialization']!=initialization_spec(arm)):
            raise ValueError('unexpected or duplicate confirmation cell')
        contract={k:meta[k] for k in screen['matched_contract']}
        if contract!=screen['matched_contract']:
            raise ValueError('confirmation must match screen source, data and learning configuration')
        cell={'seed':meta['seed'],'bundle_id':meta['bundle_id'],'run':path.parent.name,
              'initialization':meta['initialization'],'reports':{},
              'learning_curve':[{k:p[k] for k in ('steps','phase','return_pct','change_pp','mean_max_drawdown_pct','mean_costs')}
                                for p in latest_points(meta['validation_checkpoints'])]}
        for phase,filename in [('selected','selected-validation.json'),('final','validation.json')]:
            report=path.parent/filename;results=json.loads(report.read_text())
            if any(r['split']!='validation' for r in results.values()):raise ValueError('validation reports required')
            baselines={k:r['baselines']['buy_hold']['metrics'] for k,r in results.items()}
            if digest(baselines)!=screen['validation_baseline_sha256']:raise ValueError('baseline panel mismatch')
            cell[phase]=group_metrics(results,stocks)
            cell[phase]['declining_stocks']=declining_stock_summary(results,stocks)
            cell['reports'][phase]=checksum(report)
            score=meta['selected_score'] if phase=='selected' else meta['validation_checkpoints'][-1]['score']
            if not math.isclose(cell[phase]['score'],score,abs_tol=1e-10):raise ValueError('checkpoint score mismatch')
        if meta['validation_checkpoints'][-1]['phase']!='post_update':raise ValueError('post-update endpoint required')
        cells[arm][seed]=cell
    result={'protocol_sha256':screen['protocol_sha256'],'candidate':candidate,'cells':cells,
            'confirmation':confirmation_gate(cells,candidate,protocol),'matched_contract':screen['matched_contract']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['confirmation'],indent=2))


if __name__=='__main__':main()

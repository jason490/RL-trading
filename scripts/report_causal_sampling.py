"""Report the declared factorial screen or paired causal-sampling confirmation."""
import argparse
import json
import math
from pathlib import Path

from rl_stocks.core.contracts import digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.initialization import group_metrics,screen_gate,confirmation_gate
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.training.curves import latest_points
from rl_stocks.training.initialization import initialization_spec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path('configs/experiments/causal_sampling.json'))
    p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--screen',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());preflight=json.loads(a.preflight.read_text())
    if not preflight['passed'] or preflight['protocol_sha256']!=digest(protocol):raise ValueError('matching data preflight required')
    state=json.loads(a.state.read_text())
    if state['status']!='complete' or not state['cleanup_verified']:raise ValueError('complete tasks and verified cleanup required')
    universe=json.loads(Path(protocol['universe']).read_text());stocks={i['symbol'] for i in universe['instruments'] if i['kind']=='stock'}
    screen=json.loads(a.screen.read_text()) if a.screen else None
    if screen and (screen['protocol_sha256']!=digest(protocol) or not screen['screen']['passed']):raise ValueError('matching promoted screen required')
    arms=(protocol['reference_arm'],screen['screen']['promoted_candidate']) if screen else protocol['arms']
    cells={arm:{'11':screen['cells'][arm]} for arm in arms} if screen else {}
    contract=screen['matched_contract'] if screen else None
    baseline_hash=screen['validation_baseline_sha256'] if screen else None
    keys=('source_hash','git','simulation','versions','resolved_parameters','reward_config','evaluation_every','initialization')
    for item in state['completed'].values():
        paths=list((Path(item['root'])/'runs').glob('*/metadata.json'))
        if len(paths)!=1:raise ValueError('one run per task required')
        path=paths[0];meta=json.loads(path.read_text());identity=meta['dataset'];mode=meta['start_sampling']['mode'];seed=str(meta['seed'])
        matches=[arm for arm,c in protocol['arm_config'].items() if (c['dataset'],c['start_sampling'])==(identity,mode)]
        if len(matches)!=1:raise ValueError('unexpected dataset/mode combination')
        arm=matches[0]
        if (arm not in arms or meta['seed'] not in ((23,37) if screen else (11,)) or meta['status']!='complete'
                or meta['smoke'] or meta['algorithm']!='ppo' or meta['device']!='cpu'
                or meta['actual_steps']!=50000 or meta['requested_steps']!=50000
                or meta['initialization']!=initialization_spec('wide') or meta['git']['dirty']!=''):
            raise ValueError('unexpected sampler study cell')
        current={key:meta[key] for key in keys}
        if contract is not None and contract!=current:raise ValueError('unmatched source or learning configuration')
        contract=current
        if digest(meta['start_sampling'])!=preflight['pool_sha256'][identity+':'+mode]:raise ValueError('start pool differs from preflight')
        counts=meta['sampling_diagnostics']
        if counts!=json.loads((path.parent/'training-sampling.json').read_text()) or counts['transitions']!=50000:
            raise ValueError('training sampler diagnostics mismatch')
        cell={'seed':meta['seed'],'dataset':identity,'sampling':counts,'pool_sha256':digest(meta['start_sampling']),
              'bundle_id':meta['bundle_id'],'run':path.parent.name,'selected_step':meta['selected_step'],'reports':{},
              'learning_curve':[{k:r[k] for k in ('steps','phase','return_pct','change_pp','mean_max_drawdown_pct','mean_costs')}
                                for r in latest_points(meta['validation_checkpoints'])]}
        for phase,filename in [('selected','selected-validation.json'),('final','validation.json')]:
            report=path.parent/filename;results=json.loads(report.read_text())
            if any(r['split']!='validation' for r in results.values()):raise ValueError('validation-only reports required')
            baselines={key:r['baselines']['buy_hold']['metrics'] for key,r in results.items()}
            if baseline_hash is not None and digest(baselines)!=baseline_hash:raise ValueError('baseline panel mismatch')
            baseline_hash=digest(baselines);cell[phase]=group_metrics(results,stocks)
            cell[phase]['declining_stocks']=declining_stock_summary(results,stocks);cell['reports'][phase]=checksum(report)
            expected=meta['selected_score'] if phase=='selected' else meta['validation_checkpoints'][-1]['score']
            if not math.isclose(expected,cell[phase]['score'],abs_tol=1e-10):raise ValueError('report/checkpoint mismatch')
        if meta['validation_checkpoints'][-1]['phase']!='post_update':raise ValueError('final post-update point required')
        if screen:
            if seed in cells[arm]:raise ValueError('duplicate confirmation cell')
            cells[arm][seed]=cell
        else:
            if arm in cells:raise ValueError('duplicate screen cell')
            cells[arm]=cell
    gate=confirmation_gate(cells,arms[1],protocol) if screen else screen_gate(cells,protocol)
    output={'protocol_sha256':digest(protocol),'matched_contract':contract,'validation_baseline_sha256':baseline_hash,
            'cells':cells,'confirmation' if screen else 'screen':gate,'objective_achieved':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    figure=make_subplots(rows=3,cols=1,shared_xaxes=True,subplot_titles=('Validation return (%)','Incremental change (pp)','Mean maximum drawdown (%)'))
    colors=('#2563eb','#059669','#dc2626','#a855f7')
    for arm,color in zip(arms,colors):
        runs=cells[arm].values() if screen else [cells[arm]]
        for cell in runs:
            points=cell['learning_curve'];name=f"{arm}, seed {cell['seed']}"
            dash={11:'solid',23:'dash',37:'dot'}[cell['seed']]
            for row,key in ((1,'return_pct'),(2,'change_pp'),(3,'mean_max_drawdown_pct')):
                figure.add_trace(go.Scatter(x=[p['steps'] for p in points],y=[p[key] for p in points],name=name,
                    legendgroup=name,showlegend=row==1,line={'color':color,'dash':dash},mode='lines+markers'),row=row,col=1)
    figure.add_hline(y=100*sum(b['net_return'] for b in baselines.values())/len(baselines),row=1,col=1,line_dash='dash',annotation_text='Same-symbol holding')
    figure.update_layout(height=1000,title='Causal start sampling: development evidence only; final points are post-update')
    figure.write_html(str(a.output.with_suffix('.html')),include_plotlyjs=True)
    print(json.dumps(gate,indent=2))


if __name__=='__main__':main()

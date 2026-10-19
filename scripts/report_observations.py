"""Strict matched execution-state study reporting with two independent controls."""
import argparse
import json
import math
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.environment.observations.state import observation_spec
from rl_stocks.training.initialization import initialization_spec
from rl_stocks.training.curves import latest_points
from rl_stocks.reporting.initialization import group_metrics,screen_gate,confirmation_gate
from rl_stocks.evaluation.direction import declining_stock_summary


def dual_screen(cells,protocol):
    comparisons={};candidate=protocol.get('promotion_candidate','execution_full')
    for reference in protocol['promotion_references']:
        pair={arm:cells[arm] for arm in (reference,candidate)}
        comparisons[reference]=screen_gate(pair,{**protocol,'arms':list(pair),'reference_arm':reference})
    passed=all(c['passed'] for c in comparisons.values())
    return {'passed':passed,'promoted_candidate':candidate if passed else None,'controls':comparisons}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path('configs/experiments/execution_observations.json'))
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--screen',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());state=json.loads(a.state.read_text())
    if state['status']!='complete' or not state['cleanup_verified']:raise ValueError('complete study and verified cleanup required')
    architecture=protocol.get('study_kind')=='policy_architecture'
    candidate=protocol.get('promotion_candidate','execution_full')
    previous=json.loads(a.screen.read_text()) if a.screen else None
    if previous and (previous['protocol_sha256']!=digest(protocol) or not previous['screen']['passed']):
        raise ValueError('matching passing screen required')
    universe=json.loads(Path(protocol['universe']).read_text())
    stocks={i['symbol'] for i in universe['instruments'] if i['kind']=='stock'}
    cells={arm:{'11':previous['cells'][arm]} for arm in protocol['arms']} if previous else {}
    contract=previous['matched_contract'] if previous else None
    baseline=previous['validation_baseline_sha256'] if previous else None
    for item in state['completed'].values():
        paths=list((Path(item['root'])/'runs').glob('*/metadata.json'))
        if len(paths)!=1:raise ValueError('one run per task required')
        path=paths[0];m=json.loads(path.read_text());arm=(m.get('policy_architecture',{}).get('variant') if architecture else m.get('observation',{}).get('profile'));seed=str(m['seed'])
        if (arm not in protocol['arms'] or m.get('observation')!=observation_spec('execution_full' if architecture else arm)
            or m['status']!='complete' or m['smoke'] or m['algorithm']!='ppo' or m['device']!='cpu'
            or m['actual_steps']!=50000 or m['requested_steps']!=50000 or m['dataset']!=protocol['dataset']
            or m['initialization']!=initialization_spec('wide') or m['start_sampling']!={'mode':'legacy_uniform'}
            or m['seed'] not in ((23,37) if previous else (11,))):raise ValueError('unexpected execution-observation cell')
        current={k:m[k] for k in ('dataset','source_hash','git','simulation','versions','resolved_parameters','reward_config','evaluation_every','initialization','start_sampling')}
        if architecture:
            from rl_stocks.rl.ppo.architecture import architecture_spec
            from rl_stocks.rl.registry.factory import resolved_parameters
            overrides={'n_steps':protocol['n_steps'],'batch_size':protocol['batch_size']}
            if (m.get('policy_architecture')!=architecture_spec(arm,protocol['market_features'])
                or m['resolved_parameters']!=resolved_parameters('ppo',False,overrides,arm,protocol['market_features'])):
                raise ValueError('policy architecture/resolved-parameter mismatch')
            current['resolved_parameters']=resolved_parameters('ppo',False,overrides)
        if current['git']['dirty'] or contract is not None and current!=contract:raise ValueError('source/configuration mismatch')
        contract=current
        cell={'seed':m['seed'],'run':path.parent.name,'bundle_id':m['bundle_id'],'observation':m['observation'],'reports':{},'policy_architecture':m.get('policy_architecture')}
        cell['learning_curve']=[{k:r[k] for k in ('steps','phase','return_pct','change_pp','mean_max_drawdown_pct','mean_costs')} for r in latest_points(m['validation_checkpoints'])]
        for phase,file in [('selected','selected-validation.json'),('final','validation.json')]:
            rp=path.parent/file;results=json.loads(rp.read_text())
            if any(r['split']!='validation' for r in results.values()):raise ValueError('development validation only')
            bh=digest({k:r['baselines']['buy_hold']['metrics'] for k,r in results.items()})
            if baseline is not None and baseline!=bh:raise ValueError('buy-and-hold panel mismatch')
            baseline=bh;cell[phase]=group_metrics(results,stocks)
            cell[phase]['declining_stocks']=declining_stock_summary(results,stocks)
            cell['reports'][phase]=checksum(rp)
            expected=m['selected_score'] if phase=='selected' else m['validation_checkpoints'][-1]['score']
            if not math.isclose(cell[phase]['score'],expected,abs_tol=1e-10):raise ValueError('checkpoint/report score mismatch')
        if m['validation_checkpoints'][-1]['phase']!='post_update':raise ValueError('post-update final required')
        if previous:
            if seed in cells[arm]:raise ValueError('duplicate cell')
            cells[arm][seed]=cell
        else:
            if arm in cells:raise ValueError('duplicate cell')
            cells[arm]=cell
    if previous:
        result={r:confirmation_gate({r:cells[r],candidate:cells[candidate]},candidate,
                 {**protocol,'reference_arm':r}) for r in protocol['promotion_references']}
        passed=all(r['relative_improvement_passed'] for r in result.values())
        verdict={'passed':passed,'controls':result}
    else:verdict=dual_screen(cells,protocol)
    output={'protocol_sha256':digest(protocol),'protocol':protocol,'matched_contract':contract,
            'validation_baseline_sha256':baseline,'cells':cells,
            'confirmation' if previous else 'screen':verdict,'objective_achieved':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,subplot_titles=('Development return (%)','Checkpoint change (pp)','Mean maximum drawdown (%)'))
    for arm in protocol['arms']:
        runs=cells[arm].values() if previous else (cells[arm],)
        for cell in runs:
            for row,key in ((1,'return_pct'),(2,'change_pp'),(3,'mean_max_drawdown_pct')):
                points=cell['learning_curve'];label=f"{arm} seed {cell['seed']}"
                fig.add_trace(go.Scatter(x=[r['steps'] for r in points],y=[r[key] for r in points],name=label,legendgroup=label,showlegend=row==1,mode='lines+markers'),row=row,col=1)
    fig.update_layout(height=1000,title=protocol['name']+': matched development study')
    fig.write_html(str(a.output.with_suffix('.html')),include_plotlyjs=True)
    print(json.dumps(verdict,indent=2))


if __name__=='__main__':main()

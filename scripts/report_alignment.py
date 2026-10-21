"""Report strict matched recurrent sizing-alignment screens after verified owned shutdown."""
import argparse,json,math,tarfile,tempfile
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.runtime.colab.jobs import alignment_plan
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.training.recurrent_alignment import protocol,PROTOCOL_SHA256,validate_payload,modes,validate_execution_counts,validate_execution_report
from rl_stocks.training.curves import latest_points
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.reporting.history import screen_gate,render_report
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.evaluation.report_contract import validate_development_report
from rl_stocks.environment.execution.simulator import SimulationConfig


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new report output required')
    state=json.loads(a.state.read_text());spec=protocol()
    if state['status']!='complete' or not state['cleanup_verified']:raise ValueError('complete screen and verified cleanup required')
    universe=json.loads(Path('configs/universes/sector_research.json').read_text());stocks={i['symbol'] for i in universe['instruments'] if i['kind']=='stock'}
    cells={};contract=baseline=None;all_symbols=None
    for item in state['completed'].values():
        root=Path('artifacts/colab')/item['session'];artifact=root/'retrieved/artifacts';paths=list((artifact/'runs').glob('*/metadata.json'))
        if len(paths)!=1:raise ValueError('one run per task required')
        path=paths[0];m=json.loads(path.read_text());arm=next((a for a in spec['arms'] if modes(a)==m.get('action_execution')),None)
        if arm in cells or arm not in spec['arms'] or m['seed']!=11 or m['smoke']:raise ValueError('one screen seed per arm required')
        archive=root/'artifacts.tar.gz'
        if archive.exists():validate_archive(archive,alignment_plan(arm),spec['dataset'])
        else:
            # Duplicate transport archives may be removed after verified retrieval.
            # Revalidate the retained complete extraction through a temporary tar.
            with tempfile.TemporaryDirectory(prefix='sac-state-verify-') as work:
                packed=Path(work)/'retained.tar'
                with tarfile.open(packed,'w') as t:t.add(artifact,arcname='artifacts')
                validate_archive(packed,alignment_plan(arm),spec['dataset'])
        data=validate_payload(read_dataset(Path('data/datasets')/m['dataset']))
        current={k:m[k] for k in ('source_hash','git','simulation','versions','reward_config','evaluation_every','initialization','policy_architecture')}
        if current['git']['dirty'] or contract is not None and current!=contract:raise ValueError('matched source/settings required')
        contract=current;cell=dict(dataset=m['dataset'],run=path.parent.name,bundle_id=m['bundle_id'],seed=11,observation=m['observation'],reports={},sampling=m['sampling_diagnostics'])
        cell['initialization_record']=m['alignment_initialization']
        cell['training_execution']=validate_execution_counts(m['training_execution'],arm,spec['steps_per_cell'])
        cell['final_tensor_sha256']=m['alignment_final']['policy_tensor_sha256']
        from rl_stocks.rl.registry.factory import CLASSES
        from rl_stocks.training.recurrent_alignment import model_record,validate_record
        for name,key in [('latest.zip','alignment_final'),('best.zip','alignment_selected')]:
            restored=model_record(CLASSES[m['algorithm']].load(path.parent/name,device='cpu'))
            validate_record(restored,arm,selected_steps=m['selected_step'] if name=='best.zip' else None)
            if restored!=m[key]:raise ValueError('restored actual optimizer/configuration mismatch')
        from rl_stocks.training.diversity import validate_counts
        cell['sampling_summary']=validate_counts(m['sampling_diagnostics'],'expanded',False,spec['steps_per_cell'])
        cell['learning_curve']=latest_points(m['validation_checkpoints'])
        for phase,filename in [('selected','selected-validation.json'),('final','validation.json')]:
            rp=path.parent/filename;results=json.loads(rp.read_text());validate_execution_report(results,arm);validate_development_report(results,data,SimulationConfig(**m['simulation']));symbols={v['symbol'] for v in results.values()}
            expected={e['symbol'] for e in data['episodes'].values() if e['split']=='validation'}
            if symbols!=expected or all_symbols is not None and symbols!=all_symbols or any(v['split']!='validation' for v in results.values()):raise ValueError('same declared validation panel required')
            all_symbols=symbols;bh=digest({k:v['baselines']['buy_hold'] for k,v in results.items()})
            if baseline is not None and baseline!=bh:raise ValueError('matched buy-and-hold paths required')
            baseline=bh;cell[phase]=group_metrics(results,stocks)
            cell[phase]['declining_stocks']=declining_stock_summary(results,stocks)
            cell['reports'][phase]=dict(sha256=checksum(rp),file=filename)
        if (not math.isclose(cell['selected']['score'],m['selected_score'],rel_tol=0.,abs_tol=1e-12)
            or not math.isclose(m['selected_score'],max(v['score'] for v in m['validation_checkpoints']),rel_tol=0.,abs_tol=1e-12)
            or not math.isclose(cell['final']['score'],cell['learning_curve'][-1]['score'],rel_tol=0.,abs_tol=1e-12)):
            raise ValueError('selected/final report scores disagree with recorded checkpoints')
        cells[arm]=cell
    if set(cells)!=set(spec['arms']):raise ValueError('complete recurrent-state arms required')
    initial_equal=all(c['initialization_record']==cells['adaptive']['initialization_record'] for c in cells.values())
    if not initial_equal:raise ValueError('initial recurrent tensors/configurations differ')
    if any(c['sampling']!=cells['adaptive']['sampling'] for c in cells.values()):
        raise ValueError('actual symbol/group sampling counts must match')
    if cells['adaptive']['final_tensor_sha256']!=cells['posthoc_unit']['final_tensor_sha256']:
        raise ValueError('adaptive/posthoc final training weights differ; control invalid')
    verdict=screen_gate(cells,spec)
    result=dict(initial_networks_identical=initial_equal,protocol_sha256=PROTOCOL_SHA256,cells=cells,screen=verdict,matched_contract=contract,
        validation_baseline_sha256=baseline,objective_achieved=False,test_evaluated=False,
        confirmation_allocated=False,training_performed_by_report=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    render_report(result,a.output.with_suffix('.html'),title='Recurrent sizing alignment')
    print(json.dumps(verdict,indent=2))


if __name__=='__main__':main()

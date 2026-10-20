"""Report fixed temperature screens after verified owned shutdown."""
import argparse,json,math,tarfile,tempfile
from stable_baselines3 import SAC
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.runtime.colab.jobs import temperature_plan
from rl_stocks.runtime.colab.artifacts import validate_archive
from rl_stocks.training.sac_temperature import protocol,PROTOCOL_SHA256,validate_payload
from rl_stocks.training.curves import latest_points
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.reporting.history import render_report
from rl_stocks.reporting.temperature import gate,model_equal
from rl_stocks.evaluation.direction import declining_stock_summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new report output required')
    state=json.loads(a.state.read_text());spec=protocol()
    if state['status']!='complete' or not state['cleanup_verified']:raise ValueError('complete screen and verified cleanup required')
    universe=json.loads(Path('configs/universes/sector_research.json').read_text());stocks={i['symbol'] for i in universe['instruments'] if i['kind']=='stock'}
    cells={};contract=baseline=None;all_symbols=None;fingerprints=None;default_path=None
    for item in state['completed'].values():
        root=Path('artifacts/colab')/item['session'];artifact=root/'retrieved/artifacts';paths=list((artifact/'runs').glob('*/metadata.json'))
        if len(paths)!=1:raise ValueError('one run per task required')
        path=paths[0];m=json.loads(path.read_text());arm=next((k for k,v in spec['entropy_coefficient'].items() if v==m['resolved_parameters']['ent_coef']),None)
        if arm in cells or arm not in spec['arms'] or m['seed']!=11 or m['smoke']:raise ValueError('one screen seed per arm required')
        archive=root/'artifacts.tar.gz'
        if archive.exists():validate_archive(archive,temperature_plan(arm),spec['dataset'])
        else:
            # Duplicate transport archives may be removed after verified retrieval.
            # Revalidate the retained complete extraction through a temporary tar.
            with tempfile.TemporaryDirectory(prefix='temperature-verify-') as work:
                packed=Path(work)/'retained.tar'
                with tarfile.open(packed,'w') as t:t.add(artifact,arcname='artifacts')
                validate_archive(packed,temperature_plan(arm),spec['dataset'])
        data=validate_payload(read_dataset(Path('data/datasets')/m['dataset']))
        current={k:m[k] for k in ('dataset','source_hash','git','simulation','versions','resolved_parameters','reward_config','evaluation_every','initialization','policy_architecture')}
        current['resolved_parameters']={**current['resolved_parameters'],'ent_coef':'declared_temperature_arm'}
        if current['git']['dirty'] or contract is not None and current!=contract:raise ValueError('matched source/settings required')
        contract=current
        initial=m['temperature_initialization']['network_sha256']
        if fingerprints is not None and initial!=fingerprints:raise ValueError('initial actor/critic fingerprints differ')
        fingerprints=initial
        if arm=='default':default_path=path.parent
        cell=dict(dataset=m['dataset'],run=path.parent.name,bundle_id=m['bundle_id'],seed=11,observation=m['observation'],reports={},sampling=m['sampling_diagnostics'])
        cell['learning_curve']=latest_points(m['validation_checkpoints'])
        for phase,filename in [('selected','selected-validation.json'),('final','validation.json')]:
            rp=path.parent/filename;results=json.loads(rp.read_text());symbols={v['symbol'] for v in results.values()}
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
    verdict=gate(cells,spec)
    old_state=json.loads(Path('.runtime/sac-state-screen-state.json').read_text())
    if old_state['status']!='complete' or not old_state['cleanup_verified']:raise ValueError('completed prior control required')
    old_root=Path(old_state['completed']['sac-state-execution_full-11']['root'])
    old_path=next((old_root/'runs').glob('*/metadata.json'));old=json.loads(old_path.read_text())
    fresh=json.loads((default_path/'metadata.json').read_text())
    score_points=lambda m:[(r['steps'],r['phase'],r['score']) for r in m['validation_checkpoints']]
    reproduction=dict(configuration_equal=all(old[k]==fresh[k] for k in ('dataset','simulation','versions','resolved_parameters','reward_config','observation','initialization','start_sampling','actual_steps')),checkpoint_scores_exact=score_points(old)==score_points(fresh),models={})
    for filename in ('best.zip','latest.zip'):
        reproduction['models'][filename]=model_equal(SAC.load(old_path.parent/filename,device='cpu'),SAC.load(default_path/filename,device='cpu'))
    reproduction['passed']=reproduction['configuration_equal'] and reproduction['checkpoint_scores_exact'] and all(reproduction['models'].values())
    if not reproduction['passed']:
        verdict.update(passed=False,promoted_candidate=None,control_reproduction_failed=True)

    result=dict(control_reproduction=reproduction,initial_networks_identical=True,protocol_sha256=PROTOCOL_SHA256,cells=cells,screen=verdict,matched_contract=contract,
        validation_baseline_sha256=baseline,objective_achieved=False,test_evaluated=False,
        confirmation_allocated=False,training_performed_by_report=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    render_report(result,a.output.with_suffix('.html'),title='SAC initial temperature')
    print(json.dumps(verdict,indent=2))


if __name__=='__main__':main()

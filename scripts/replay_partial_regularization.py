"""Recover aligned paths from the fixed retained80k checkpoint; no selection."""
import argparse,json,math
from pathlib import Path
import torch
from stable_baselines3 import SAC
from diagnose_training_fit import weights
from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.runner import evaluate,selection_score
from rl_stocks.evaluation.report_contract import validate_development_report
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.inference.bundle import checksum
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.reporting.participation import diagnose_groups
from rl_stocks.reporting.html import report
from rl_stocks.training.sac_regularization import validate_payload,optimizer_record,validate_final_optimizer_record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.artifacts.exists():raise ValueError('new output paths required')
    source=Path('results/sac-regularization-2026-10-20/interruption-summary.json')
    summary=json.loads(source.read_text());cell=summary['records']['weight_decay']
    if summary['status']!='interrupted_no_promotion' or not summary['cleanup_verified'] or cell['last_retrieved_steps']!=80000:
        raise ValueError('fixed recovered80k interruption required')
    model_path=Path(next(p for p in summary['source_sha256'] if cell['run'] in Path(p).parts and Path(p).name=='latest-checkpoint.zip'))
    metadata_path=model_path.parent/'metadata.json';metadata=json.loads(metadata_path.read_text())
    data_path=Path('data/datasets')/metadata['dataset']
    protected={str(p):checksum(p) for p in (source,model_path,metadata_path,data_path/'dataset.json')}
    for p in (model_path,metadata_path):
        if protected[str(p)]!=summary['source_sha256'][str(p)]:raise ValueError('retained source mismatch')
    data=validate_payload(read_dataset(data_path));normalizer=digest(data['normalizer'])
    torch.set_num_threads(1);torch.set_flush_denormal(False)
    model=SAC.load(model_path,device='cpu');model.policy.set_training_mode(False)
    if model.num_timesteps!=80000 or metadata['observation']['profile']!='execution_full':raise ValueError('fixed partial full-state checkpoint required')
    before=weights(model);updates=model._n_updates;optimizer=optimizer_record(model)
    validate_final_optimizer_record(optimizer,'weight_decay')
    config=SimulationConfig(**metadata['simulation'])
    print('Replaying fixed80k checkpoint on26 development episodes; no fitting.',flush=True)
    results=evaluate(model,data,'validation',config,'execution_full')
    validate_development_report(results,data,config)
    score=selection_score(results);expected=cell['curve'][-1]['score']
    print('Replay score / recorded score:',score,expected,flush=True)
    score_matches=math.isclose(score,expected,rel_tol=0,abs_tol=1e-10)
    if weights(model)!=before or model._n_updates!=updates or optimizer_record(model)!=optimizer or digest(data['normalizer'])!=normalizer:
        raise ValueError('weights/alpha/optimizer/preprocessing mutated')
    stocks={i['symbol'] for i in data['manifest']['universe_config']['instruments'] if i['kind']=='stock'}
    args.artifacts.mkdir(parents=True)
    report(results,args.artifacts/'validation-80k.html','Retained80k model replay; score parity not established; not final100k')
    if not all(checksum(Path(p))==h for p,h in protected.items()):raise ValueError('source files mutated')
    result=dict(status='intermediate_replay_score_tolerance_failed' if not score_matches else 'intermediate_checkpoint_replayed',steps=80000,phase='on_step',run=cell['run'],
        replay_score=score,recorded_curve_score=expected,curve_score_reproduced=score_matches,curve_score_absolute_tolerance=1e-10,score_difference=score-expected,mean_equity_difference_at_initial_capital=(score-expected)*config.initial_equity,groups=group_metrics(results,stocks),
        declining_stocks=declining_stock_summary(results,stocks),participation=diagnose_groups(results,stocks),
        weights_alpha_optimizer_normalization_unchanged=True,source_files_unchanged=True,source_sha256=protected,
        weights_sha256=before,normalization_sha256=normalizer,script_sha256=checksum(Path(__file__)),
        report_sha256=checksum(args.artifacts/'validation-80k.json'),test_evaluated=False,checkpoint_selected=False,
        training_performed=False,colab_allocated=False,numerical_training_setting_changed=False,
        original_screen_assessable=False,objective_achieved=False,
        interpretation='Deterministic replay of fixed intermediate checkpoint for diagnostic paths. Score tolerance failures remain failures; no relaxed acceptance or parity claim. Not100k final, no promotion or reinterpretation of incomplete study. Repeated2021-2023 development only; future decline cohort reporting-only.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Recovered80k score:',100*score,'joint positive short/total:',result['declining_stocks']['positive_policy_and_short_pnl_count'],flush=True)


if __name__=='__main__':main()

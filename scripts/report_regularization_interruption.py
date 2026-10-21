"""Revalidate retained incomplete SAC screen; never fit, promote or restart."""
import argparse
import json,tarfile,hashlib,csv
from pathlib import Path
import torch
from stable_baselines3 import SAC
from rl_stocks.inference.bundle import checksum
from rl_stocks.training.sac_regularization import protocol,PROTOCOL_SHA256,validate_final_optimizer_record,optimizer_record
from rl_stocks.training.diversity import validate_counts
from rl_stocks.evaluation.report_contract import validate_development_report
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.core.contracts import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('new output directory required')
    args.output.mkdir(parents=True)
    spec=protocol();out=args.output;state=json.load(open('.runtime/regularization-screen-state.json'));ledger=json.load(open('.runtime/usage-ledger.json'));active=ledger['attempts'][-1]
    assert state['status']=='failed' and state['cleanup_verified'] and ledger['budget_stopped']
    assert all(c['status']=='stopped' for c in active['children'].values())
    sid='rl-stocks-cpu-b30e88282e11';archive=Path('artifacts/colab')/sid/'checkpoint.tar.gz';protected={str(archive):checksum(archive)}
    extraction=archive.parent/'interruption-retrieved'
    if not extraction.exists():
     with tarfile.open(archive) as t:t.extractall(extraction,filter='data')
    control=Path(state['completed']['regularization-control-11']['root'])
    roots={'control':control,'weight_decay':extraction/'artifacts'}
    data=read_dataset(Path('data/datasets')/spec['dataset']);stocks={i['symbol'] for i in data['manifest']['universe_config']['instruments'] if i['kind']=='stock'}
    records={};rows=[];torch.set_num_threads(1)
    for arm,root in roots.items():
     path=next((root/'runs').glob('*/metadata.json'));m=json.loads(path.read_text());protected[str(path)]=checksum(path)
     assert m['dataset']==spec['dataset'] and m['study_protocol_sha256']==PROTOCOL_SHA256
     assert m['git']['revision']=='6792d2bac1d2c920007a82a7c301f247684a19da'
     curve=[json.loads(x) for x in (path.parent/'learning-curve.jsonl').read_text().splitlines()]
     step=curve[-1]['steps'];records[arm]=dict(run=path.parent.name,source_revision=m['git']['revision'],status=m['status'],last_retrieved_steps=step,curve=curve,reports={},restored={})
     for filename in (['latest.zip','best.zip'] if arm=='control' else ['latest-checkpoint.zip','best.zip']):
      p=path.parent/filename;protected[str(p)]=checksum(p);model=SAC.load(p,device='cpu');r=optimizer_record(model);validate_final_optimizer_record(r,arm)
      assert all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
      records[arm]['restored'][filename]=dict(num_timesteps=model.num_timesteps,n_updates=model._n_updates,optimizer=r,sha256=protected[str(p)])
     assert records[arm]['restored']['latest.zip' if arm=='control' else 'latest-checkpoint.zip']['num_timesteps']==step
     for phase,filename in [('selected','selected-validation.json'),('final' if arm=='control' else 'saved_validation_unaligned_to_latest','validation.json')]:
      p=path.parent/filename;protected[str(p)]=checksum(p);report=json.loads(p.read_text());validate_development_report(report,data,SimulationConfig(**m['simulation']))
      records[arm]['reports'][phase]=dict(metrics=group_metrics(report,stocks),declining_stocks=declining_stock_summary(report,stocks),sha256=protected[str(p)])
     for v in curve:rows.append(dict(arm=arm,steps=v['steps'],return_pct=v['score']*100,phase=v['phase']))
     counts=json.loads((path.parent/'training-sampling.json').read_text());records[arm]['sampling']=validate_counts(counts,'expanded',False,step)
    assert all(checksum(Path(p))==v for p,v in protected.items())
    result=dict(status='interrupted_no_promotion',protocol_sha256=PROTOCOL_SHA256,screen_assessable=False,reason='Metering unavailable; protective budget stop. Candidate100k post-update final missing;80k is not substituted for final.',records=records,source_sha256=protected,source_files_unchanged=True,cleanup_verified=True,budget_stopped=True,automatic_restart=False,test_evaluated=False,training_performed_by_report=False,objective_achieved=False,project_units=ledger['consumed'],remaining_units=150-ledger['consumed'],study_incremental_units=ledger['consumed']-36.26389526975882,screen_measured_units=active['measured'],screen_estimated_units=active['estimated'],post_recovery_account=dict(balance=349.97,rate=0,assignments=0))
    (out/'interruption-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (out/'learning-curves.csv').open('w') as f:
     w=csv.DictWriter(f,fieldnames=['arm','steps','return_pct','phase']);w.writeheader();w.writerows(rows)
    print('Restored and validated complete control plus incomplete80k candidate; no promotion or restart.')


if __name__=='__main__':main()

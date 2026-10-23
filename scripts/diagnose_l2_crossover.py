"""Fixed four-case saved-policy crossover; diagnostic only, never train/select."""
import argparse,json,subprocess
from pathlib import Path
import torch
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.runner import evaluate
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.evaluation.report_contract import validate_development_report
from rl_stocks.inference.bundle import checksum
from rl_stocks.inference.crossover import RecurrentCrossover,require_native_parity
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.reporting.optimizer import optimizer_fingerprint
from rl_stocks.reporting.html import report
from rl_stocks.rl.registry.factory import CLASSES
from rl_stocks.training.recurrent_regularization import protocol as source_protocol,validate_payload,model_record,validate_record,identify_arm

PROTOCOL_SHA256='4b33f924813e9dd0fe337477b9ca80925ea4afe4e6cf53ba3f0adf6b1ece3e1f'
SOURCE_FILES=('configs/experiments/l2_crossover.json','scripts/diagnose_l2_crossover.py',
    'src/rl_stocks/inference/crossover.py','src/rl_stocks/evaluation/runner.py')


def protocol():
    p=json.loads(Path(SOURCE_FILES[0]).read_text())
    if digest(p)!=PROTOCOL_SHA256:raise ValueError('fixed crossover protocol changed')
    return p


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.artifacts.exists():raise ValueError('new output/artifact paths required')
    spec=protocol();root=Path.cwd().resolve();protected={}
    for name in SOURCE_FILES:
        if subprocess.check_output(['git','show','HEAD:'+name])!=(root/name).read_bytes():raise ValueError('commit fixed crossover source before replay')
    inventory=json.loads(Path(spec['source_inventory']).read_text())
    def protect(path,accepted=False):
        path=Path(path);key=str(path.resolve().relative_to(root));value=checksum(path)
        if accepted and inventory.get(key)!=value:raise ValueError('accepted retained-source hash mismatch')
        protected[key]=value
    for name in [*SOURCE_FILES,spec['source_state'],spec['source_summary'],spec['source_inventory']]:protect(name)
    protect(spec['source_summary'],True)
    state=json.loads(Path(spec['source_state']).read_text());accepted=json.loads(Path(spec['source_summary']).read_text())
    if state['status']!='complete' or not state['cleanup_verified'] or accepted['test_evaluated'] or digest(source_protocol())!=spec['source_protocol_sha256']:
        raise ValueError('accepted cleaned unchanged source required')
    cell=accepted['cells'][spec['source_arm']];item=state['completed']['l2-l2-11'];run=Path(item['root'])/'runs'/cell['run']
    protect(run/'metadata.json',True);m=json.loads((run/'metadata.json').read_text())
    if identify_arm(m)!='l2' or m['smoke'] or m['seed']!=11 or m['status']!='complete' or m['bundle_id']!=cell['bundle_id'] or m['actual_steps']!=100000:raise ValueError('exact accepted L2 model required')
    data_path=Path('data/datasets')/spec['dataset'];protect(data_path/'dataset.json',True);data=validate_payload(read_dataset(data_path))
    normalizer=digest(data['normalizer']);models={};before={};optim={};saved={}
    torch.set_num_threads(1)
    for phase,filename,key in [('selected','best.zip','l2_selected'),('final','latest.zip','l2_final')]:
        path=run/filename;protect(path,True);model=CLASSES[m['algorithm']].load(path,device='cpu');models[phase]=model
        before[phase]=model_record(model);optim[phase]=optimizer_fingerprint(model)
        validate_record(before[phase],'l2',selected_steps=m['selected_step'] if phase=='selected' else None)
        if before[phase]!=m[key]:raise ValueError('restored source model mismatch')
        rp=run/cell['reports'][phase]['file'];protect(rp,True)
        if checksum(rp)!=cell['reports'][phase]['sha256']:raise ValueError('accepted native report hash mismatch')
        saved[phase]=json.loads(rp.read_text())
    config=SimulationConfig(**m['simulation'])
    if config!=SimulationConfig():raise ValueError('unchanged default simulator required')
    stocks={i['symbol'] for i in data['manifest']['universe_config']['instruments'] if i['kind']=='stock'}
    records={};native={};baseline=None;a.artifacts.mkdir(parents=True)
    for case in spec['cases']:
        mixed=RecurrentCrossover(models,case['direction'],case['protection'])
        results=evaluate(mixed,data,'validation',config,spec['observation_profile'],spec['exposure_mode'])
        results=json.loads(canonical(results));validate_development_report(results,data,config)
        bh=digest({k:v['baselines']['buy_hold'] for k,v in results.items()})
        if baseline is not None and baseline!=bh:raise ValueError('matched holding paths required')
        baseline=bh
        if case['name'] in spec['native_controls']:
            require_native_parity(saved[case['direction']],results,atol=spec['native_tolerance']['atol'],rtol=spec['native_tolerance']['rtol']);native[case['name']]=True
        elif set(native)!=set(spec['native_controls']):raise ValueError('both native controls must pass before mixed replay')
        directory=a.artifacts/case['name'];directory.mkdir();report(results,directory/'validation.html','Diagnostic component crossover; not selected/promoted')
        records[case['name']]=dict(**case,metrics=group_metrics(results,stocks),declining_stocks=declining_stock_summary(results,stocks),derived_report_sha256=checksum(directory/'validation.json'))
        print(case['name'],'mean_return',round(100*records[case['name']]['metrics']['score'],4),flush=True)
    if (any(model_record(model)!=before[k] or optimizer_fingerprint(model)!=optim[k] for k,model in models.items())
        or digest(data['normalizer'])!=normalizer or any(checksum(root/k)!=v for k,v in protected.items())):raise ValueError('source/model/optimizer/normalization changed')
    result=dict(protocol_sha256=PROTOCOL_SHA256,records=records,native_controls_reproduced=native,source_sha256=protected,
        model_records=before,optimizer_sha256=optim,normalizer_sha256=normalizer,holding_sha256=baseline,
        source_files_unchanged=True,weights_optimizer_normalization_unchanged=True,training_performed=False,
        checkpoint_selected=False,hybrid_promoted=False,colab_allocated=False,provider_requested=False,test_evaluated=False,
        objective_achieved=False,interpretation=spec['interpretation'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()

"""Four fixed causal sizing replays; no fitting or checkpoint selection."""
import argparse,gzip,json
from pathlib import Path
import torch
from rl_stocks.core.contracts import digest,canonical
from rl_stocks.core.provenance import repository_state
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.evaluation.runner import evaluate
from rl_stocks.evaluation.report_contract import validate_development_report
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.inference.bundle import checksum,FrozenPolicy
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.reporting.history import screen_gate
from rl_stocks.rl.registry.factory import CLASSES
from replay_directions import policy_fingerprint,verify_reference,reference_differences

PROTOCOL_SHA256='d9754c21f3679d7a22139a19518ec4d9ac20674fd2a2408a22403bdae26c6c68'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists():raise ValueError('new output directory required')
    git=repository_state(Path.cwd())
    if git['dirty']:raise ValueError('clean committed replay source required')
    spec=json.loads(Path('configs/experiments/recurrent_unit_sizing.json').read_text())
    if digest(spec)!=PROTOCOL_SHA256:raise ValueError('fixed sizing protocol changed')
    state=json.loads(Path('.runtime/recurrent-state-screen-state.json').read_text())
    if state['status']!='complete' or not state['cleanup_verified']:raise ValueError('verified source shutdown required')
    accepted=json.loads(Path('results/recurrent-state-2026-10-21/summary.json').read_text())['cells'][spec['source_arm']]
    item=state['completed']['recurrent-state-recurrent_masked-11'];root=Path(item['root'])
    paths=list((root/'runs').glob('*/metadata.json'))
    if len(paths)!=1:raise ValueError('one accepted run required')
    path=paths[0];m=json.loads(path.read_text())
    if (path.parent.name!=accepted['run'] or m['git']['revision']!=spec['source_revision'] or m['git']['dirty']
            or m['algorithm']!=spec['algorithm'] or m['seed']!=11 or m['actual_steps']!=100000
            or m['dataset']!=spec['dataset'] or m['observation']['profile']!=spec['observation_profile']):
        raise ValueError('fixed accepted source contract required')
    data=read_dataset(Path('data/datasets')/spec['dataset'])
    from rl_stocks.training.recurrent_state import validate_payload
    validate_payload(data)
    if digest(data['normalizer'])!=spec['normalizer_sha256']:raise ValueError('immutable normalization required')
    feed=data['manifest']['provenance']['capabilities']['feed'];bundle=root/'frozen'/m['bundle_id']
    frozen=FrozenPolicy(bundle,data['schema'],feed)
    if checksum(bundle/'model.zip')!=checksum(path.parent/'best.zip'):raise ValueError('selected frozen hash mismatch')
    universe=json.loads(Path('configs/universes/sector_research.json').read_text());stocks={i['symbol'] for i in universe['instruments'] if i['kind']=='stock'}
    protected={str(p):checksum(p) for p in [path,path.parent/'best.zip',path.parent/'latest.zip',bundle/'manifest.json',bundle/'model.zip']}
    normalizer=digest(data['normalizer']);models={};controls={};parity={};cells={arm:{} for arm in spec['arms']}
    a.output.mkdir(parents=True);torch.set_num_threads(1)
    def write(value):(a.output/'summary.json').write_text(json.dumps(value,indent=2)+'\n')
    for phase,name in [('selected','best.zip'),('final','latest.zip')]:
        report=path.parent/accepted['reports'][phase]['file'];protected[str(report)]=checksum(report)
        if checksum(report)!=accepted['reports'][phase]['sha256']:raise ValueError('accepted path hash mismatch')
        original=json.loads(report.read_text());model=CLASSES[spec['algorithm']].load(path.parent/name,device='cpu');model.policy.set_training_mode(False)
        models[phase]=(model,policy_fingerprint(model),model._n_updates,model.num_timesteps)
        print('Replay unrestricted',phase,flush=True)
        replay=evaluate(model,data,'validation',SimulationConfig(**m['simulation']),spec['observation_profile'])
        validate_development_report(replay,data,SimulationConfig(**m['simulation']))
        differences=reference_differences(replay,original)
        try:parity[phase]=verify_reference(replay,original,spec['parity_tolerance'])
        except (ValueError,AssertionError) as exc:
            write(dict(status='reference_gate_failed',phase=phase,error=str(exc),differences=differences,protocol_sha256=PROTOCOL_SHA256,candidate_evaluated=False,objective_achieved=False,training_performed=False,test_evaluated=False));raise
        parity[phase]['differences']=differences;controls[phase]=replay
    for arm in spec['arms']:
        for phase in spec['phases']:
            model,before,updates,steps=models[phase]
            print('Replay',arm,phase,flush=True)
            results=controls[phase] if arm=='control' else evaluate(model,data,'validation',SimulationConfig(**m['simulation']),spec['observation_profile'],exposure_mode=spec['exposure_mode'][arm])
            validate_development_report(results,data,SimulationConfig(**m['simulation']))
            metrics=group_metrics(results,stocks);metrics['declining_stocks']=declining_stock_summary(results,stocks);cells[arm][phase]=metrics
            with gzip.open(a.output/f'{arm}-{phase}.json.gz','wt') as f:json.dump(results,f,separators=(',',':'))
            if policy_fingerprint(model)!=before or model._n_updates!=updates or model.num_timesteps!=steps:raise ValueError('model changed during replay')
    if digest(data['normalizer'])!=normalizer or any(checksum(Path(p))!=v for p,v in protected.items()):raise ValueError('source/preprocessing changed')
    verdict=screen_gate(cells,spec)
    write(dict(status='complete',protocol_sha256=PROTOCOL_SHA256,replay_source=git,parity=parity,cells=cells,screen=verdict,source_hashes=protected,weights_updates_normalization_unchanged=True,training_performed=False,test_evaluated=False,colab_allocated=False,objective_achieved=False))
    print(json.dumps(verdict,indent=2),flush=True)


if __name__=='__main__':main()

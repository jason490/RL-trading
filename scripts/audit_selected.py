"""One held-out audit after the complete validation selection is committed."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean,stdev
import subprocess


def committed_decision(path, repository):
    repository=Path(repository).resolve();path=Path(path).resolve()
    relative=path.relative_to(repository)
    committed=subprocess.run(['git','show','HEAD:'+relative.as_posix()],cwd=repository,
                             check=True,capture_output=True).stdout
    if committed!=path.read_bytes():raise ValueError('commit the validation selection before testing')
    decision=json.loads(committed)
    if decision.get('test_evaluated') is not False:
        raise ValueError('expected a frozen validation-only decision')
    return decision


def audit(decision_path, roots, output, repository):
    import torch
    from rl_stocks.core.contracts import digest
    from rl_stocks.data.generation.datasets import read_dataset
    from rl_stocks.environment.execution.simulator import SimulationConfig
    from rl_stocks.evaluation.runner import evaluate,selection_score
    from rl_stocks.inference.bundle import FrozenPolicy,checksum
    from rl_stocks.reporting.html import report
    torch.set_num_threads(1)
    decision=committed_decision(decision_path,repository)
    family=decision['selected_family']
    expected_seeds=decision['protocol']['seeds']
    expected_runs=len(decision['protocol']['datasets'])*len(decision['protocol']['algorithms'])*len(expected_seeds)
    if (family['seeds']!=expected_seeds or len(family['bundle_ids'])!=len(expected_seeds)
            or len(decision.get('runs',[]))!=expected_runs):
        raise ValueError('complete matched study and all predeclared family seeds required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    dataset_id=decision['protocol']['datasets'][family['dataset']]
    dataset=read_dataset(Path(repository)/'data/datasets'/dataset_id)
    available={p.name:p for root in map(Path,roots) for p in (root/'frozen').iterdir()}
    result={'status':'running','decision_hash':digest(decision),'family':family,
            'candidate':decision['candidate'],'dataset':dataset_id,'device':'cpu',
            'test_evaluated':True,'independent_symbol_episodes':True,'seeds':[]}
    target=output/'audit.json'
    def save():target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    try:
        for seed,bundle_id in zip(family['seeds'],family['bundle_ids']):
            bundle=available[bundle_id]
            hashes={p.name:checksum(p) for p in bundle.iterdir()}
            policy=FrozenPolicy(bundle,dataset['schema'],
                dataset['manifest']['provenance']['capabilities']['feed'],device='cpu')
            if policy.manifest['metadata']['seed']!=seed or policy.manifest['metadata']['algorithm']!=family['algorithm']:
                raise ValueError('selected family/bundle mismatch')
            before=asdict(policy.normalizer)
            results=evaluate(policy._model,dataset,'test',SimulationConfig(**policy.manifest['metadata']['simulation']))
            report(results,output/f'test-seed-{seed}.html',f'Held-out audit: frozen {family["algorithm"]}, seed {seed}')
            if before!=asdict(policy.normalizer) or hashes!={p.name:checksum(p) for p in bundle.iterdir()}:
                raise RuntimeError('inference modified frozen artifacts or normalization')
            result['seeds'].append({'seed':seed,'bundle_id':bundle_id,'mean_net_return':selection_score(results),
                'symbols':{k:{'symbol':r['symbol'],'policy':r['metrics'],
                    'baselines':{name:b['metrics'] for name,b in r['baselines'].items()}}
                    for k,r in results.items()}})
            save()
        scores=[r['mean_net_return'] for r in result['seeds']]
        result.update(status='complete',mean_net_return=mean(scores),seed_standard_deviation=stdev(scores),
            mean_buy_hold_return=mean(r['baselines']['buy_hold']['net_return'] for r in result['seeds'][0]['symbols'].values()),
            normalizers_and_bundles_unchanged=True)
        save()
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__);save();raise
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('decision');parser.add_argument('roots',nargs='+')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(audit(args.decision,args.roots,args.output,Path.cwd()))

"""Verify dataset panels and causal start-pool identities before allocation."""
import argparse
import json
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.environment.sampling.regimes import CausalStarts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path('configs/experiments/causal_sampling.json'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());panels={};records={};pool_hashes={}
    for identity in sorted({c['dataset'] for c in protocol['arm_config'].values()}):
        dataset=read_dataset(Path('data/datasets')/identity)
        if dataset['manifest']['unverified_override'] or dataset['manifest']['included_splits']!=['train','validation']:
            raise ValueError('verified development-only dataset required')
        training=[e for e in dataset['episodes'].values() if e['split']=='train']
        validation={k:e for k,e in dataset['episodes'].items() if e['split']=='validation'}
        panels[identity]=digest(validation)
        if len(validation)!=26:raise ValueError('original 26-symbol panel required')
        records[identity]={'train_symbols':len(training),'training_rows':sum(len(e['bars']) for e in training),
                           'validation_rows':sum(len(e['bars']) for e in validation.values()),
                           'validation_sha256':panels[identity],'normalizer_sha256':digest(dataset['normalizer']),
                           'names':dataset['names'],'schema':dataset['schema'],
                           'feed':dataset['manifest']['provenance']['capabilities']['feed'],'pools':{}}
        for mode in ('ready_uniform','downtrend_balanced'):
            summary=CausalStarts(training,mode).summary
            records[identity]['pools'][mode]=summary
            pool_hashes[identity+':'+mode]=digest(summary)
    if len(set(panels.values()))!=1:raise ValueError('raw validation episodes differ across datasets')
    if len({(tuple(r['names']),r['schema'],r['feed']) for r in records.values()})!=1:
        raise ValueError('matched feature and provider contracts required')
    value={'passed':True,'protocol_sha256':digest(protocol),'datasets':records,'pool_sha256':pool_hashes,
           'training_performed':False,'retired_test_evaluated':False,'sessions_allocated':0}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2)+'\n')
    print('Verified matching validation panels and four training-only start pools; no allocation.')


if __name__=='__main__':main()

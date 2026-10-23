"""Build a checksummed recent training suffix, preserving common development."""
import argparse,json
from pathlib import Path
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.data.generation.training_history import recent_subset,validate_subset

SOURCE='cdb93a33cbfd12c85a0d7ea50af8e0e8843356407b1fbcc8124efa0792d2223b'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new evidence path required')
    root=Path('data/datasets');source=read_dataset(root/SOURCE)
    recent=recent_subset(source);identity=digest(recent);dest=root/identity;dest.mkdir(exist_ok=True)
    write_once(dest/'dataset.json',canonical(recent));validate_subset(source,read_dataset(dest))
    assert digest(read_dataset(root/SOURCE))==SOURCE
    result=dict(passed=True,source_dataset=SOURCE,recent_dataset=identity,recent_start='2016-01-01',
        normalizer_sha256=digest(source['normalizer']),schema=recent['schema'],
        development_sha256=recent['manifest']['training_history']['development_sha256'],
        training_symbols=len(recent['manifest']['training_history']['training_symbols']),
        training_rows=sum(len(e['bars']) for e in recent['episodes'].values() if e['split']=='train'),
        validation_rows=sum(len(e['bars']) for e in recent['episodes'].values() if e['split']=='validation'),
        retained_source_features_bars_atr_exact=True,development_exact=True,normalization_unchanged=True,
        source_unchanged=True,training_performed=False,test_evaluated=False,colab_allocated=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

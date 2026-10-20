"""Audit cached sources and build fixed-preprocessing expansion; no network/fitting."""
import argparse,json
from pathlib import Path
from rl_stocks.core.contracts import canonical,digest
from rl_stocks.core.files import write_once
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.inference.bundle import checksum
from rl_stocks.training.diversity import ORIGINAL,EXPANDED,audit_sources,fixed_preprocessing_expansion


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('new audit output required')
    sources={key:Path('data/datasets')/identity for key,identity in [('original',ORIGINAL),('expanded',EXPANDED)]}
    protected={str(path/'dataset.json'):checksum(path/'dataset.json') for path in sources.values()}
    original=read_dataset(sources['original']);expanded=read_dataset(sources['expanded'])
    audit=audit_sources(original,expanded)
    if audit['training']['original']['episodes']!=26 or audit['training']['expanded']['episodes']!=111:
        raise ValueError('fixed26 versus111 source required')
    payload=fixed_preprocessing_expansion(original,expanded);identity=digest(payload)
    destination=Path('data/datasets')/identity;destination.mkdir(exist_ok=True)
    write_once(destination/'dataset.json',canonical(payload))
    assert read_dataset(destination)['normalizer']==original['normalizer']
    assert all(checksum(Path(p))==h for p,h in protected.items())
    audit.update(source_sha256=protected,fixed_preprocessing_dataset=identity,
        fixed_preprocessing_payload_sha256=checksum(destination/'dataset.json'),source_files_unchanged=True,
        training_performed=False,colab_allocated=False,test_evaluated=False,objective_achieved=False,
        implementation_sha256={p:checksum(Path(p)) for p in ['scripts/prepare_sac_diversity.py','src/rl_stocks/training/diversity.py']})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print('Audited original/expanded sources; fixed-preprocessing dataset',identity)


if __name__=='__main__':main()

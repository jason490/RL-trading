"""Verify tree dependencies/data/construction; never fit a local model."""
import argparse,json,re,sys
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.inference.bundle import checksum
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.training.supervised.spec import SIGNAL_DATASET,validate_payload
from rl_stocks.training.supervised.forest_spec import protocol,make_forest,verify_forest_environment

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    config=protocol();versions=verify_forest_environment();data=validate_payload(read_dataset(Path('data/signals')/SIGNAL_DATASET))
    model=make_forest(11);prefix=make_forest(11,True)
    if hasattr(model,'estimators_') or hasattr(prefix,'estimators_'):raise ValueError('preflight must not fit models')
    parse=lambda path:dict(re.findall(r'^([a-zA-Z0-9_-]+)==([^ \\]+)',Path(path).read_text(),re.M))
    old_path=Path('.runtime/requirements-before-trees.lock');changes=None
    if old_path.exists():
        old=parse(old_path);new=parse('requirements.lock');changes={k:[v,new.get(k)] for k,v in old.items() if new.get(k)!=v}
        if changes:raise ValueError('unrelated existing dependency pins changed')
    result={'passed':True,'protocol_sha256':digest(config),'signal_dataset':SIGNAL_DATASET,
        'python':sys.version.split()[0],'versions':versions,'requirements_lock_sha256':checksum(Path('requirements.lock')),
        'prior_package_pin_changes':changes,'constructor_parameters':model.get_params(),'prefix_tree_count':prefix.n_estimators,
        'fit_rows':sum(r['fold']=='fit' for r in data['rows']),'probe_rows':sum(r['fold']=='probe' for r in data['rows']),
        'model_fitted':False,'colab_allocated':False,'project_validation_used':False,'test_evaluated':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Verified compatible optional dependencies, immutable folds and constructor; no model fitted.')

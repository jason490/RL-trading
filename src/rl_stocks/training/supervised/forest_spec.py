"""Fixed tree feasibility protocol and optional dependency acceptance."""
import importlib.metadata
import json
from pathlib import Path
from rl_stocks.core.contracts import digest
from rl_stocks.training.dependencies import verify_environment

TREE_VERSIONS={'scikit-learn':'1.9.1','scipy':'1.17.1','joblib':'1.6.0','threadpoolctl':'3.7.0'}
PROTOCOL_SHA256='49aa716bb364a97c1d34631eef49d049fc3c7a16afcd4933501e2a52a4999eb8'


def protocol():
    value=json.loads((Path(__file__).resolve().parents[4]/'configs/experiments/forest_signal.json').read_text())
    if digest(value)!=PROTOCOL_SHA256:raise ValueError('forest protocol drift')
    return value


def verify_forest_environment():
    core=verify_environment()
    try:actual={name:importlib.metadata.version(name) for name in TREE_VERSIONS}
    except importlib.metadata.PackageNotFoundError as error:raise RuntimeError('install the hash-locked trees extra before forest research') from error
    if actual!=TREE_VERSIONS:raise RuntimeError('forest dependency versions differ from reviewed lock')
    return {'core':core,'trees':actual}


def make_forest(seed,prefix=False):
    if type(seed) is not int or seed not in (11,23,37) or type(prefix) is not bool or prefix and seed!=11:
        raise ValueError('unsupported forest seed or prefix mode')
    verify_forest_environment()
    from sklearn.ensemble import RandomForestRegressor
    params=dict(protocol()['models']['forest']);params.pop('class');params.pop('checkpoint_selection');params['random_state']=seed
    if prefix:params['n_estimators']=2
    return RandomForestRegressor(**params)

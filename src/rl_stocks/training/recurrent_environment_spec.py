"""Fixed cached-data environment smoke and causal prefix-view contract."""
from copy import deepcopy
import json
from pathlib import Path

from rl_stocks.core.contracts import digest
from rl_stocks.training.recurrent_loop import LoopConfig, validate_dataset

PROTOCOL_SHA256 = '9f376211f2f9578cbe0e0757fdaf8f7c0da6295e73c6575a7d4094d69843e995'
ROOT = Path(__file__).resolve().parents[3]


def protocol():
    value = json.loads((ROOT/'configs/experiments/recurrent_environment_smoke.json').read_text())
    if digest(value) != PROTOCOL_SHA256:
        raise ValueError('fixed recurrent environment smoke required')
    config = LoopConfig(**value['loop'])
    if config.maximum_updates != value['expected_updates'] or 3*config.maximum_updates != value['maximum_adam_steps']:
        raise ValueError('fixed actual transition/update bounds required')
    return value


def validate_view(view):
    spec = protocol()
    normalizer, _ = validate_dataset(view, digest(view))
    manifest = view['manifest']['recurrent_environment_smoke']
    expected = dict(protocol_sha256=PROTOCOL_SHA256, source_dataset=spec['source_dataset'],
                    train_prefix_rows=spec['view']['train_prefix_rows'],
                    validation_prefix_rows=spec['view']['validation_prefix_rows'],
                    normalization='Original frozen training-only statistics; no refit',
                    performance_evaluation=False)
    if manifest != expected or digest(view['normalizer']) != spec['normalizer_sha256']:
        raise ValueError('fixed source/prefix/preprocessing smoke identity required')
    if (view['manifest']['universe'] != spec['symbols']
            or view['manifest']['included_splits'] != ['train', 'validation']
            or view['manifest']['unverified_override'] is not False):
        raise ValueError('fixed smoke universe/split/coverage contract required')
    for split, key in [('train', 'train_prefix_rows'), ('validation', 'validation_prefix_rows')]:
        episodes = [ep for ep in view['episodes'].values() if ep['split'] == split]
        if sorted(ep['symbol'] for ep in episodes) != spec['symbols']:
            raise ValueError('exact smoke symbol membership required')
        rows = spec['view'][key]
        for ep in episodes:
            if (len(ep['bars']) != rows or len(ep['features']) != rows or len(ep['atr']) != rows
                    or any(b['symbol'] != ep['symbol'] for b in ep['bars'])):
                raise ValueError('exact aligned smoke prefix rows required')
            for features in ep['features']:
                normalizer.transform(features, view['names'])
    return view


def build_view(source):
    spec = protocol()
    if (digest(source) != spec['source_dataset'] or digest(source['normalizer']) != spec['normalizer_sha256']
            or source['manifest']['included_splits'] != ['train', 'validation']
            or source['manifest']['unverified_override'] is not False
            or any(ep['split'] not in ('train', 'validation') for ep in source['episodes'].values())):
        raise ValueError('original verified train/development source and preprocessing required')
    episodes = {}
    for key, ep in source['episodes'].items():
        if ep['symbol'] not in spec['symbols']:
            continue
        rows = spec['view']['train_prefix_rows' if ep['split'] == 'train' else 'validation_prefix_rows']
        if len(ep['bars']) < rows:
            raise ValueError('fixed source prefix unavailable; no substitution')
        item = {k: deepcopy(v) for k, v in ep.items() if k not in ('bars', 'features', 'atr')}
        for field in ('bars', 'features', 'atr'):
            item[field] = deepcopy(ep[field][:rows])
        episodes[key] = item
    view = dict(schema=source['schema'], names=deepcopy(source['names']),
                normalizer=deepcopy(source['normalizer']), episodes=episodes,
                manifest=dict(universe=spec['symbols'], included_splits=['train', 'validation'],
                              unverified_override=False, provenance=deepcopy(source['manifest']['provenance']),
                              source_manifest_sha256=digest(source['manifest']),
                              recurrent_environment_smoke=dict(protocol_sha256=PROTOCOL_SHA256,
                              source_dataset=spec['source_dataset'],
                              train_prefix_rows=spec['view']['train_prefix_rows'],
                              validation_prefix_rows=spec['view']['validation_prefix_rows'],
                              normalization='Original frozen training-only statistics; no refit',
                              performance_evaluation=False)))
    validate_view(view)
    return view

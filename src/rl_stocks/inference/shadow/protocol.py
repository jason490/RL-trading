"""Content-addressed campaign registration; no validation-driven replacement."""
from dataclasses import asdict
import hashlib
import json
import re
from pathlib import Path
from rl_stocks.core.contracts import canonical, digest, utcnow
from rl_stocks.core.files import write_once
from rl_stocks.core.provenance import repository_state
from rl_stocks.data.storage.immutable import load
from rl_stocks.data.validation.bars import timestamp
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.inference.bundle import FrozenPolicy
from rl_stocks.inference.shadow.journal import transaction, get, set_value


def implementation_id():
    root = Path(__file__).resolve().parents[2]
    paths = [*root.glob('features/**/*.py'), *root.glob('environment/**/*.py'),
             *root.glob('risk/**/*.py'), *root.glob('inference/**/*.py'),
             root/'data/generation/datasets.py', root/'data/validation/bars.py']
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(paths)})


def register(config, output, *, now=None, policy_loader=FrozenPolicy):
    now = timestamp(now or utcnow())
    if set(config) != {'name', 'snapshot', 'arms', 'review_sessions', 'max_age_seconds'}:
        raise ValueError('campaign configuration keys mismatch')
    if not isinstance(config['review_sessions'], int) or config['review_sessions'] < 2:
        raise ValueError('review requires at least two sessions')
    if not 60 <= config['max_age_seconds'] <= 86400:
        raise ValueError('invalid freshness limit')
    bars, provenance = load(config['snapshot'])
    if any(timestamp(b.retrieved_time) > now or timestamp(b.available_time) > now for b in bars):
        raise ValueError('future bootstrap data')
    if not config['arms'] or len({a['name'] for a in config['arms']}) != len(config['arms']):
        raise ValueError('unique campaign arms required')
    if len({(a['family'], a['seed']) for a in config['arms']}) != len(config['arms']):
        raise ValueError('duplicate seed within a family')
    arms = []
    common = None
    for arm in config['arms']:
        if set(arm) != {'name', 'family', 'seed', 'bundle'}:
            raise ValueError('arm keys mismatch')
        if not all(isinstance(arm[k], str) and re.fullmatch(r'[A-Za-z0-9_-]+', arm[k]) for k in ('name', 'family')):
            raise ValueError('invalid arm/family name')
        path = Path(arm['bundle']).resolve()
        manifest = json.loads((path/'manifest.json').read_text())
        cap = manifest['data_manifest']['provenance']['capabilities']
        policy_loader(path, manifest['feature_schema'], cap['feed'])
        if manifest['metadata']['seed'] != arm['seed']:
            raise ValueError('arm seed mismatch')
        data = manifest['data_manifest']
        sim = asdict(SimulationConfig(**manifest['metadata']['simulation']))
        identity = (tuple(data['universe']), data['snapshot'], sim, cap)
        if common is not None and identity != common:
            raise ValueError('arms must share universe, bootstrap, costs and feed')
        common = identity
        if data['snapshot'] != Path(config['snapshot']).name:
            raise ValueError('bootstrap snapshot differs from training context')
        arms.append({**arm, 'bundle': str(path), 'bundle_id': path.name,
                     'manifest_sha256': hashlib.sha256(canonical(manifest)).hexdigest()})
    symbols = common[0]
    required = sorted({b.symbol for b in bars})
    contract = {
        'version': 1, 'name': config['name'], 'registered_at': now.isoformat(),
        'not_before': max(now, timestamp('2026-10-01T00:00:00+00:00')).isoformat(),
        'snapshot': str(Path(config['snapshot']).resolve()),
        'snapshot_id': Path(config['snapshot']).name, 'arms': arms,
        'symbols': symbols, 'required_symbols': required, 'simulation': common[2],
        'feed_contract': common[3], 'review_sessions': config['review_sessions'],
        'max_age_seconds': config['max_age_seconds'], 'implementation': implementation_id(),
        'git': repository_state(Path.cwd()), 'mode': 'shadow-no-orders',
        'selection': 'All registered arms retained; no forward checkpoint selection.',
        'review': {'minimum_mean_excess_return_pp': 0., 'minimum_positive_seeds': 2,
                   'maximum_mean_drawdown_worsening_pp': 2.,
                   'decision': 'Research review only; no automatic paper/live eligibility.'},
        'interpretation': 'Separate stock accounts; no investable portfolio claim. '
                          'Retrospective surviving universe; historical membership/delistings unverified.'}
    identity = digest(contract)
    target = Path(output)/identity
    target.mkdir(parents=True, exist_ok=True)
    with transaction(target/'journal.sqlite3') as db:
        if get(db, 'campaign') not in (None, identity):
            raise ValueError('journal campaign mismatch')
        write_once(target/'protocol.json', canonical(contract))
        set_value(db, 'campaign', identity)
    return target


def read_protocol(root):
    root = Path(root)
    value = json.loads((root/'protocol.json').read_text())
    if digest(value) != root.name:
        raise ValueError('campaign protocol changed')
    if value['implementation'] != implementation_id():
        raise ValueError('shadow implementation changed; review and register a new campaign')
    return value

"""Selected/final reference control evaluation; original native selection stays fixed."""
import gzip
import hashlib
import json
from pathlib import Path
import time

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.core.provenance import repository_state
from rl_stocks.data.generation.datasets import read_dataset
from rl_stocks.evaluation.canonical_diagnostic import inference_only
from rl_stocks.evaluation.canonical_task import evaluate_canonical_training
from rl_stocks.evaluation.direction import declining_stock_summary
from rl_stocks.inference.canonical_bundle import FrozenCanonicalTaskPolicy, freeze_canonical_task
from rl_stocks.reporting.html import report
from rl_stocks.reporting.initialization import group_metrics
from rl_stocks.runtime.colab.canonical_market_probe import file_sha
from rl_stocks.training.task_conditioning import protocol as base_protocol
from rl_stocks.training.task_reference_assets import PROTOCOL_SHA256, protocol

ASSETS_SHA256 = 'b12c1796280f86e5d848fd0372eee0cc639213e091bfafe70b13d8b25ec11292'


def load_assets(root):
    root = Path(root)
    spec = protocol(root)
    path = root / 'results/task-conditioning-2026-10-31/reference-control-assets.json'
    if file_sha(path) != ASSETS_SHA256:
        raise ValueError('fixed complete physical control admission required')
    receipt = json.loads(path.read_text())
    if (receipt['protocol_sha256'] != PROTOCOL_SHA256
            or set(receipt['controls']) != set(spec['controls'])
            or file_sha(root/receipt['artifact_path']) != receipt['artifact_sha256']):
        raise ValueError('physical control admission identity mismatch')
    return receipt


def reuse_engineering(root, reference_bundle_id):
    """Use the accepted full paths, never an unbound scalar performance number."""
    root = Path(root)
    protocol(root)  # Pins the exact engineering receipt before reading results.
    accepted = json.loads((root / 'results/reference-inference-2026-11-02/market-cross-machine-acceptance.json').read_text())
    folder = root / 'artifacts/colab' / accepted['session']
    meta = folder / 'reference-probe.json'
    path = folder / 'market-paths.json.gz'
    if (file_sha(meta) != accepted['actual_remote_report_sha256']
            or file_sha(path) != accepted['retrieved_paths_sha256']):
        raise ValueError('accepted engineering paths changed')
    metadata = json.loads(meta.read_text())
    if metadata['reference_bundle_id'] != reference_bundle_id:
        raise ValueError('engineering point belongs to a different reference actor')
    with gzip.open(path, 'rb') as stream:
        raw = stream.read(128*1024*1024+1)
    if len(raw) > 128*1024*1024 or hashlib.sha256(raw).hexdigest() != accepted['results_sha256']:
        raise ValueError('complete accepted engineering result checksum mismatch')
    return json.loads(raw), str(path.relative_to(root)), accepted['results_sha256']


def store_report(results, destination):
    """Keep interactive HTML and lossless compressed canonical paths, not duplicates."""
    destination = Path(destination)
    report(results, destination, 'Reference control — fixed training-selected/final snapshot')
    raw = canonical(results)
    paths = destination.with_suffix('.paths.json.gz')
    with paths.open('xb') as stream:
        stream.write(gzip.compress(raw, mtime=0))
    sidecar = destination.with_suffix('.json')
    if digest(json.loads(sidecar.read_text())) != hashlib.sha256(raw).hexdigest():
        raise ValueError('generated report sidecar differs from evaluated paths')
    if gzip.decompress(paths.read_bytes()) != raw:
        raise ValueError('generated compressed paths mismatch')
    sidecar.unlink()  # Generated duplicate; canonical full paths remain losslessly retained.
    return paths


def validate_matrix(cells, assets):
    if set(cells) != set(assets['controls']):
        raise ValueError('complete two-control reference matrix required')
    holding = None
    for role, phases in cells.items():
        if set(phases) != {'selected', 'final'}:
            raise ValueError('selected and final reference controls required')
        for phase, cell in phases.items():
            expected = assets['controls'][role]
            if (cell['role'] != role or cell['phase'] != phase or cell['period'] != 'training-window'
                    or cell['protocol_sha256'] != PROTOCOL_SHA256 or cell['seed'] != 11
                    or cell['steps'] != (expected['selected_step'] if phase == 'selected' else 100000)
                    or cell['native_bundle_id'] != expected['frozen_bundles'][phase]
                    or cell['selection_execution'] != 'retained-native-training-curve'
                    or cell['performance_execution'] != 'reference-actor-v1'
                    or cell['episodes'] != 26 or cell['predictions'] != 19136):
                raise ValueError('fixed native-selected reference performance contract required')
            if holding is None:
                holding = cell['holding_path_sha256']
            elif cell['holding_path_sha256'] != holding:
                raise ValueError('reference controls require identical holding paths')
    return holding


def run(root, output):
    root, output = Path(root), Path(output)
    spec = protocol(root)
    assets = load_assets(root)
    source = repository_state(root)
    if source['dirty']:
        raise ValueError('clean committed reference-control evaluation source required')
    dataset = read_dataset(root/'data/datasets'/base_protocol()['dataset'])
    stocks = set(dataset['manifest']['evaluation_symbols']) - {'SPY', 'QQQ', 'IWM', 'DIA'}
    output.mkdir(parents=True, exist_ok=False)
    cells = {}
    started = time.monotonic()
    current = None
    try:
        with inference_only():
            for role, admitted in assets['controls'].items():
                cells[role] = {}
                for phase in ('selected', 'final'):
                    current = f'{role}-{phase}'
                    print('Reference evaluation:', current, flush=True)
                    native_id = admitted['frozen_bundles'][phase]
                    native = root/'artifacts/colab'/spec['controls'][role]['session']/'retrieved/artifacts/task-screen'/f'frozen-{phase}'/native_id
                    frozen_path = freeze_canonical_task(native, output/'frozen'/role/phase)
                    frozen = FrozenCanonicalTaskPolicy(frozen_path, feature_schema=dataset['schema'],
                        feed=dataset['manifest']['provenance']['capabilities']['feed'])
                    reused = role == 'tagged_shared' and phase == 'selected'
                    point_started = time.monotonic()
                    if reused:
                        results, paths, expected_sha = reuse_engineering(root, frozen_path.name)
                        paths = root / paths
                        html = root/'artifacts/canonical-market-replay-v1/replay.html'
                    else:
                        evidence = frozen.manifest['source_manifest']['metadata']['task_contract']['type_manifest']
                        results = evaluate_canonical_training(frozen, dataset, evidence)
                        expected_sha = digest(results)
                        html = output / f'{current}.html'
                        paths = store_report(results, html)
                    if digest(results) != expected_sha or len(results) != 26 or sum(len(r['rows']) for r in results.values()) != 19136:
                        raise ValueError('complete reference performance paths required')
                    cell = {'role':role, 'phase':phase, 'period':'training-window', 'seed':11,
                        'steps':admitted['selected_step'] if phase=='selected' else 100000,
                        'protocol_sha256':PROTOCOL_SHA256, 'base_protocol_sha256':spec['base_protocol_sha256'],
                        'native_bundle_id':native_id, 'reference_bundle_id':frozen_path.name,
                        'selection_execution':'retained-native-training-curve', 'performance_execution':'reference-actor-v1',
                        'episodes':len(results), 'predictions':sum(len(r['rows']) for r in results.values()),
                        'results_sha256':expected_sha, 'metrics':group_metrics(results, stocks),
                        'short_side':declining_stock_summary(results, stocks),
                        'holding_path_sha256':digest({k:r['baselines']['buy_hold'] for k,r in results.items()}),
                        'paths_path':str(paths.relative_to(root)), 'paths_sha256':file_sha(paths),
                        'report_path':str(html.relative_to(root)), 'report_sha256':file_sha(html),
                        'reused_accepted_engineering_point':reused, 'elapsed_seconds':time.monotonic()-point_started}
                    cells[role][phase] = cell
                    (output/f'{current}.json').write_bytes(canonical(cell))
                    print(current, 'return_pct', 100*cell['metrics']['score'], flush=True)
        holding = validate_matrix(cells, assets)
        result = {'schema':'task-reference-controls-v1', 'protocol_sha256':PROTOCOL_SHA256,
            'source':source, 'physical_assets_sha256':ASSETS_SHA256, 'dataset':digest(dataset),
            'controls':cells, 'holding_path_sha256':holding,
            'new_reference_replays':3, 'reused_engineering_replays':1,
            'elapsed_seconds':time.monotonic()-started, 'training_performed':False,
            'checkpoint_reselection':False, 'development_allowed':False,
            'native_acceptance_changed':False, 'candidate_allocation_allowed':False,
            'colab_allocated':False, 'performance_promotion':False}
        (output/'result.json').write_bytes(canonical(result))
        return result
    except BaseException as error:
        (output/'failure.json').write_bytes(canonical({'schema':'task-reference-control-failure-v1',
            'protocol_sha256':PROTOCOL_SHA256, 'source':source, 'point':current,
            'error_type':type(error).__name__, 'training_performed':False, 'colab_allocated':False}))
        raise

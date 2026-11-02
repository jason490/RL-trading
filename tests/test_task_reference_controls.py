from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from rl_stocks.core.contracts import canonical
from rl_stocks.evaluation.task_reference_controls import (
    PROTOCOL_SHA256, load_assets, reuse_engineering, store_report, validate_matrix,
)


def matrix():
    assets = {'controls':{role:{'selected_step':50000, 'frozen_bundles':
        {'selected':role+'-selected', 'final':role+'-final'}} for role in ('masked_shared','tagged_shared')}}
    cells = {role:{phase:{'role':role, 'phase':phase, 'period':'training-window',
        'seed':11, 'protocol_sha256':PROTOCOL_SHA256,
        'steps':50000 if phase=='selected' else 100000,
        'native_bundle_id':value['frozen_bundles'][phase],
        'selection_execution':'retained-native-training-curve', 'performance_execution':'reference-actor-v1',
        'episodes':26, 'predictions':19136, 'holding_path_sha256':'a'*64}
        for phase in ('selected','final')} for role,value in assets['controls'].items()}
    return cells, assets


def test_fixed_control_matrix_and_shared_benchmark():
    cells, assets = matrix()
    assert validate_matrix(cells, assets) == 'a'*64


@pytest.mark.parametrize('change', ['missing', 'step', 'period', 'execution', 'holding', 'bundle'])
def test_matrix_rejects_reselection_mixed_execution_or_benchmark(change):
    cells, assets = matrix()
    cell = cells['tagged_shared']['selected']
    if change == 'missing':
        cells['tagged_shared'].pop('final')
    elif change == 'step':
        cell['steps'] = 60000
    elif change == 'period':
        cell['period'] = 'validation'
    elif change == 'execution':
        cell['performance_execution'] = 'native'
    elif change == 'holding':
        cell['holding_path_sha256'] = 'b'*64
    else:
        cell['native_bundle_id'] = 'different source'
    with pytest.raises(ValueError):
        validate_matrix(cells, assets)


def test_physical_admission_is_bound_to_producer_artifact(tmp_path, monkeypatch):
    from rl_stocks.evaluation import task_reference_controls as module
    producer = tmp_path / 'physical.json'
    producer.write_bytes(b'physical checker fixture')
    path = tmp_path / 'results/task-conditioning-2026-10-31/reference-control-assets.json'
    path.parent.mkdir(parents=True)
    data = {'protocol_sha256':PROTOCOL_SHA256, 'controls':{'masked_shared':{},'tagged_shared':{}},
        'artifact_path':'physical.json', 'artifact_sha256':hashlib.sha256(producer.read_bytes()).hexdigest()}
    path.write_bytes(canonical(data))
    monkeypatch.setattr(module, 'ASSETS_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(module, 'protocol', lambda root:{'controls':data['controls']})
    assert load_assets(tmp_path) == data
    producer.write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity'):
        load_assets(tmp_path)


def test_engineering_reuse_requires_exact_actor_and_paths(tmp_path, monkeypatch):
    from rl_stocks.evaluation import task_reference_controls as module
    monkeypatch.setattr(module, 'protocol', lambda root:{})
    result = {'fixture':{'equity':100000.}}
    raw = canonical(result)
    folder = tmp_path / 'artifacts/colab/fixture'
    folder.mkdir(parents=True)
    (folder/'reference-probe.json').write_bytes(canonical({'reference_bundle_id':'actor-a'}))
    paths = folder/'market-paths.json.gz'
    paths.write_bytes(gzip.compress(raw, mtime=0))
    accepted = {'session':'fixture',
        'actual_remote_report_sha256':hashlib.sha256((folder/'reference-probe.json').read_bytes()).hexdigest(),
        'retrieved_paths_sha256':hashlib.sha256(paths.read_bytes()).hexdigest(),
        'results_sha256':hashlib.sha256(raw).hexdigest()}
    path = tmp_path/'results/reference-inference-2026-11-02/market-cross-machine-acceptance.json'
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical(accepted))
    actual, _, _ = reuse_engineering(tmp_path, 'actor-a')
    assert actual == result
    with pytest.raises(ValueError, match='different reference actor'):
        reuse_engineering(tmp_path, 'actor-b')
    paths.write_bytes(gzip.compress(b'wrong', mtime=0))
    with pytest.raises(ValueError, match='changed'):
        reuse_engineering(tmp_path, 'actor-a')


def test_report_retains_lossless_paths_and_removes_only_generated_duplicate(tmp_path, monkeypatch):
    from rl_stocks.evaluation import task_reference_controls as module
    results = {'fixture':{'rows':[{'equity':100000.}], 'events':[]}}
    def writer(value, destination, title):
        Path(destination).write_text('<html>fixture</html>')
        Path(destination).with_suffix('.json').write_text(json.dumps(value, indent=2))
    monkeypatch.setattr(module, 'report', writer)
    html = tmp_path/'report.html'
    paths = store_report(results, html)
    assert html.exists() and not html.with_suffix('.json').exists()
    assert gzip.decompress(paths.read_bytes()) == canonical(results)
    altered = deepcopy(results)
    altered['fixture']['rows'][0]['equity'] += 1.
    def wrong_writer(value, destination, title):
        writer(altered, destination, title)
    monkeypatch.setattr(module, 'report', wrong_writer)
    with pytest.raises(ValueError, match='differs'):
        store_report(results, tmp_path/'bad.html')
    assert (tmp_path/'bad.json').exists()

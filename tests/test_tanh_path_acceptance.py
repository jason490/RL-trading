import json

import pytest

from test_tanh_paths import fixture, Frozen, Direct
from rl_stocks.core.contracts import canonical
from rl_stocks.evaluation.tanh_paths import paired_episode
from rl_stocks.evaluation.tanh_path_acceptance import verify_episode, restore


def test_complete_saved_path_restores_actions_accounting_and_metrics(tmp_path):
    data, episode = fixture()
    path = tmp_path / "path.jsonl"
    proof = paired_episode(episode, data, Frozen(data), Direct(), path)
    assert verify_episode(path, episode, data) == proof


@pytest.mark.parametrize("field", ["reward", "basis", "action", "next", "truncated"])
def test_saved_path_tampering_rejected(tmp_path, field):
    data, episode = fixture()
    path = tmp_path / "path.jsonl"
    paired_episode(episode, data, Frozen(data), Direct(), path)
    lines = path.read_bytes().splitlines()
    row = json.loads(lines[0])
    if field == "reward":
        row["reward"] = {"float64": float(1e-8).hex()}
    elif field == "basis":
        row["state"]["basis"] = {"float64": float(120).hex()}
    elif field == "action":
        row["action_hex"][0] = "00000000"
    elif field == "next":
        row["next_observation_hex"][0] = "3f800000"
    else:
        row["truncated"] = True
    path.write_bytes(b"\n".join([canonical(row)] + lines[1:]) + b"\n")
    with pytest.raises(ValueError):
        verify_episode(path, episode, data)


def test_incomplete_path_and_noncanonical_numbers_rejected(tmp_path):
    data, episode = fixture()
    path = tmp_path / "path.jsonl"
    paired_episode(episode, data, Frozen(data), Direct(), path)
    lines = path.read_bytes().splitlines()
    path.write_bytes(b"\n".join(lines[:-1]) + b"\n")
    with pytest.raises(ValueError, match="incomplete"):
        verify_episode(path, episode, data)
    for value in (1.2, {"float64": "nan"}, {"float64": "0x1p0"}):
        with pytest.raises(ValueError):
            restore(value)


def test_complete_directory_and_comparison_restore_actual_files(tmp_path, monkeypatch):
    import shutil
    from rl_stocks.evaluation import tanh_path_acceptance as acceptance
    from rl_stocks.core.contracts import digest
    from rl_stocks.training import trpo_smoke
    data, episode = fixture()
    data['episodes'] = {'SPY-train': episode}
    monkeypatch.setattr(trpo_smoke, 'validate_dataset', lambda d: d)
    monkeypatch.setattr(acceptance, 'training_window', lambda d, **kw: d)
    monkeypatch.setattr(acceptance, 'protocol', lambda: {
        'training_window_episodes_sha256': digest(data['episodes']),
        'cases': [{'case': 'fixture', 'bundle_id': 'fixed-bundle'}]})
    left = tmp_path / 'left'
    folder = left / 'fixture'
    folder.mkdir(parents=True)
    row = paired_episode(episode, data, Frozen(data), Direct(), folder / 'SPY-train.jsonl')
    case = {'case': 'fixture', 'bundle_id': 'fixed-bundle', 'episodes': [row],
            'decisions': row['decisions'], 'paths_digest': digest([row['path_sha256']])}
    (folder / 'result.json').write_bytes(canonical(case))
    result = {'schema': 'tanh-full-path-parity-result-v1',
              'protocol_sha256': acceptance.PROTOCOL_SHA256, 'dataset': digest(data),
              'cases': [case], 'source_dataset_unchanged': True,
              'training_performed': False, 'development_evaluated': False,
              'retired_test_access': False, 'performance_promotion': False}
    (left / 'result.json').write_bytes(canonical(result))
    right = tmp_path / 'right'
    shutil.copytree(left, right)
    proof = acceptance.compare(left, right, data)
    assert proof['exact_full_paths'] and proof['decisions'] == 34
    assert proof['runtime_locations_verified'] is False
    result['cases'][0]['episodes'][0]['metrics']['net_return'] += 1e-8
    (right / 'result.json').write_bytes(canonical(result))
    with pytest.raises(ValueError, match='restoration mismatch'):
        acceptance.compare(left, right, data)

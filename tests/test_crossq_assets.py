import pytest

from rl_stocks.training import crossq_assets as assets


def test_actual_six_archives_and_portable_source_provenance():
    result = assets.validate_assets()
    assert result['passed'] and set(result['records']) == {'11', '23', '37'}
    assert result['portable_source_snapshots_verified'] == 6
    assert not result['models_constructed'] and result['optimizer_steps'] == 0
    for arm in ['sac', 'crossq']:
        full, _ = assets.initialization_entry(arm, 11, False)
        smoke, _ = assets.initialization_entry(arm, 11, True)
        assert full == smoke and full['smoke'] is False
        proof = assets.initialization_spec(arm, 11, True)
        assert proof['smoke'] is True and proof['artifact_sha256'] == full['sha256']


@pytest.mark.parametrize('arm,seed,smoke', [('bad', 11, False), ('sac', True, False), ('crossq', 23, True), ('sac', 11, 1)])
def test_invalid_phase_or_boolean_alias_rejected(arm, seed, smoke):
    with pytest.raises(ValueError):
        assets.initialization_entry(arm, seed, smoke)


@pytest.mark.parametrize('name', ['/tmp/outside.npz', '../.env', 'configs/initializations/../../.env'])
def test_paths_cannot_escape_initialization_tree(name):
    with pytest.raises(ValueError):
        assets.portable_path(name)


def test_binding_drift_rejected_before_asset_reads(monkeypatch):
    monkeypatch.setattr(assets, 'BINDING_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='binding'):
        assets.validate_assets()


def test_portable_snapshots_validate_without_live_original_factory(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import shutil

    original = assets.ROOT
    shutil.copytree(original / 'configs/initializations/crossq_v1', tmp_path / 'configs/initializations/crossq_v1')
    for name in ['crossq_initialization_binding.json', 'distributional_initialization.json']:
        path = tmp_path / 'configs/experiments' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / 'configs/experiments' / name, path)
    manifest_path = Path('configs/initializations/distributional_v1/manifest.json')
    old = json.loads((original / manifest_path).read_text())
    files = [manifest_path] + [Path(old['artifacts'][f'sac_native-screen-{seed}']['file']) for seed in [11, 23, 37]]
    for name in files:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / name, tmp_path / name)
    changed_factory = tmp_path / 'src/rl_stocks/rl/registry/factory.py'
    changed_factory.parent.mkdir(parents=True)
    changed_factory.write_text('# deliberately different future registry\n')
    monkeypatch.setattr(assets, 'ROOT', tmp_path)
    assert assets.validate_assets()['passed']

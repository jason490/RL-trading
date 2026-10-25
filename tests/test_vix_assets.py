import json

import pytest

from rl_stocks.training import vix_assets


def test_all_fixed_vix_assets_verify_without_model_construction():
    result = vix_assets.validate_assets()
    assert result["passed"] and result["artifact_entries"] == 12
    assert result["original_common_tensors_and_cpu_rng_exact"] and result["new_columns_zero"]
    assert result["models_constructed"] is False and result["optimizer_steps"] == 0


def test_binding_edit_and_unregistered_seed_phase_reject(tmp_path, monkeypatch):
    value = vix_assets.binding()
    value["datasets"]["full"] = value["datasets"]["masked"]
    path = tmp_path / "configs/experiments/vix_context_binding.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value))
    monkeypatch.setattr(vix_assets, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        vix_assets.binding()
    with pytest.raises(ValueError):
        vix_assets.initialization_entry("full", 23, True)
    with pytest.raises(ValueError):
        vix_assets.initialization_entry("other")


@pytest.mark.parametrize("path", ["/tmp/model.npz", "../model.npz", "configs/model.npz"])
def test_only_portable_tracked_initializer_paths(path):
    with pytest.raises(ValueError):
        vix_assets._portable_path(path)

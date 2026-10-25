import json

import pytest

from rl_stocks.training import vix_spec
from rl_stocks.training.market_initialization import parameters as original_parameters


@pytest.mark.parametrize("smoke", [False, True])
def test_original_native_recurrent_parameters_unchanged(smoke):
    assert vix_spec.parameters(smoke) == original_parameters(smoke)


def test_fixed_protocol_and_input_geometry(tmp_path, monkeypatch):
    assert [vix_spec.market_features(a) + 14 for a in ["legacy", "masked", "full"]] == [21, 23, 23]
    with pytest.raises(ValueError):
        vix_spec.market_features("other")
    with pytest.raises(ValueError):
        vix_spec.parameters(1)
    protocol = vix_spec.preregistration()
    protocol["steps_per_cell"] = 200000
    path = tmp_path / "configs/experiments/vix_context_preregistration.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(protocol))
    monkeypatch.setattr(vix_spec, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        vix_spec.preregistration()

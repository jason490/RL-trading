"""Fixed residual-boosting feasibility; construction never fits data."""

import json
from pathlib import Path

from rl_stocks.core.contracts import digest

PROTOCOL_SHA256 = "7c9fa2818b68da79ffd0528be8496f4e392a5131caa0fd1fde209cf2cd169d79"


def protocol():
    path = Path(__file__).resolve().parents[4] / "configs/experiments/residual_boosting.json"
    value = json.loads(path.read_text())
    if digest(value) != PROTOCOL_SHA256:
        raise ValueError("residual-boosting protocol drift")
    return value


def make_boosting(seed, *, prefix=False):
    if type(seed) is not int or seed not in (11, 23, 37) or type(prefix) is not bool:
        raise ValueError("declared residual-boosting seed/prefix required")
    if prefix and seed != 11:
        raise ValueError("only seed11 discarded prefix allowed")
    from rl_stocks.training.supervised.forest_spec import verify_forest_environment

    verify_forest_environment()
    from sklearn.ensemble import GradientBoostingRegressor

    params = dict(protocol()["model"])
    if prefix:
        params["n_estimators"] = 2
    return GradientBoostingRegressor(**params, random_state=seed)

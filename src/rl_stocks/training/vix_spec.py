"""Pinned two-coordinate VIX research contract; no allocation or model creation."""

import json
from pathlib import Path

from rl_stocks.core.contracts import digest

ROOT = Path(__file__).resolve().parents[3]
PREREGISTRATION_SHA256 = "c9972c2ce85d0d9bad54501f1634357919217e5aee8b4b964d197f8dd6ea6008"


def preregistration():
    value = json.loads((ROOT / "configs/experiments/vix_context_preregistration.json").read_text())
    if digest(value) != PREREGISTRATION_SHA256:
        raise ValueError("fixed VIX preregistration changed")
    return value


def market_features(arm):
    if arm not in preregistration()["arms"]:
        raise ValueError("fixed legacy/masked/full VIX arm required")
    return 7 if arm == "legacy" else 9


def parameters(smoke=False):
    if type(smoke) is not bool:
        raise ValueError("explicit smoke boolean required")
    spec = preregistration()
    params = dict(spec["parameters"])
    architecture = params.pop("net_arch")
    if smoke:
        params.update(spec["smoke_parameters"])
    return dict(
        params,
        target_kl=spec["target_kl"],
        policy_kwargs=dict(
            spec["recurrent_parameters"],
            net_arch=architecture,
            optimizer_kwargs=spec["optimizer_kwargs"],
        ),
    )

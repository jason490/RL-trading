"""One fixed GAE trace comparison; no allocations, fitting or source requests."""

import json
from pathlib import Path

from rl_stocks.core.contracts import digest

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SHA256 = "e8995cc951ffb4884b18d7b5e270f74b4ee987e54200be5494f7b00d40112a59"


def protocol():
    spec = json.loads((ROOT / "configs/experiments/gae_trace_preregistration.json").read_text())
    if digest(spec) != PROTOCOL_SHA256:
        raise ValueError("fixed GAE trace protocol required")
    return spec


def parameters(arm, smoke=False):
    spec = protocol()
    if arm not in spec["arms"] or type(smoke) is not bool:
        raise ValueError("fixed GAE arm/boolean phase required")
    settings = dict(spec["parameters"])
    net_arch = settings.pop("net_arch")
    settings.update(
        gae_lambda=spec["gae_lambda"][arm],
        target_kl=spec["target_kl"],
        policy_kwargs=dict(
            spec["recurrent_parameters"],
            net_arch=net_arch,
            optimizer_kwargs=spec["optimizer_kwargs"],
        ),
    )
    if smoke:
        settings.update(spec["smoke_parameters"])
    return settings

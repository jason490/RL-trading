"""Fixed Treasury information study; no source access or policy construction."""

import json
from pathlib import Path

from rl_stocks.core.contracts import digest

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SHA256 = "b47335deed48fc2880ea3d0c0b2f80b6c4328c6a25808c95d0f96a379734a468"
BINDING_SHA256 = "0c8939febcf93f948c92b5e1c5399c209ff95776eb66f538c71b9606273d854b"


def protocol():
    value = json.loads(
        (ROOT / "configs/experiments/treasury_context_preregistration.json").read_text()
    )
    if digest(value) != PROTOCOL_SHA256:
        raise ValueError("fixed Treasury study protocol required")
    return value


def binding():
    value = json.loads((ROOT / "configs/experiments/treasury_context_binding.json").read_text())
    if digest(value) != BINDING_SHA256 or value["preregistration_sha256"] != PROTOCOL_SHA256:
        raise ValueError("fixed Treasury dataset/initializer binding required")
    return value


def parameters(arm, smoke=False):
    spec = protocol()
    if arm not in spec["arms"] or type(smoke) is not bool:
        raise ValueError("fixed Treasury arm/boolean phase required")
    settings = dict(spec["parameters"])
    architecture = settings.pop("net_arch")
    settings.update(
        target_kl=spec["target_kl"],
        policy_kwargs=dict(
            spec["recurrent_parameters"],
            net_arch=architecture,
            optimizer_kwargs=spec["optimizer_kwargs"],
        ),
    )
    if smoke:
        settings.update(spec["smoke_parameters"])
    return settings

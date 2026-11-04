from dataclasses import FrozenInstanceError
import hashlib
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from rl_stocks.core.contracts import digest
from rl_stocks.inference.canonical_actor import encode32
from rl_stocks.inference.canonical_math import CONTRACT, certified_tanh, bits32
from rl_stocks.inference.canonical_tanh_policy import CanonicalTanhPolicy, SCHEMA, OUTPUT
from rl_stocks.inference.canonical_tanh_export import export_policy
from rl_stocks.training.trpo_feasibility import create
from test_trpo_feasibility import ContractEnv


def layer(i, o):
    return {"weights": [[encode32(0.0)] * i for _ in range(o)], "biases": [encode32(0.0)] * o}


def fixture():
    body = {
        "schema": SCHEMA,
        "arithmetic": CONTRACT,
        "algorithm": "trpo",
        "input_size": 21,
        "hidden_sizes": [64, 64],
        "output_size": 4,
        "source_model_sha256": "a" * 64,
        "output_transform": OUTPUT,
        "layers": [layer(21, 64), layer(64, 64), layer(64, 4)],
    }
    for t in body["layers"]:
        t["weights"][0][0] = encode32(1.0)
    body["layers"][2]["biases"][1:] = [encode32(2.0), encode32(-2.0), encode32(0.5)]
    return {"actor": body, "sha256": digest(body)}


def test_certified_hidden_tanh_and_direct_output_clipping_and_immutability(tmp_path):
    value = fixture()
    actor = CanonicalTanhPolicy(value)
    x = [0.5] + [0.0] * 20
    got = actor.predict(x)
    assert bits32(got[0]) == bits32(certified_tanh(certified_tanh(0.5)))
    assert got[1:] == (1.0, -1.0, 0.5)  # Output must not receive a final tanh/unscale.
    value["actor"]["layers"].clear()
    view = actor.manifest
    view["actor"]["layers"].clear()
    assert actor.predict(x) == got
    with pytest.raises(FrozenInstanceError):
        actor._encoded = "bad"
    for bad in ([0.0] * 20, [0.0] * 22, [float("nan")] * 21, [True] * 21, None):
        with pytest.raises(ValueError):
            actor.predict(bad)
    path = tmp_path / "actor.json"
    path.write_text(json.dumps(actor.manifest))
    assert CanonicalTanhPolicy.load(path).predict(x) == got
    code = 'from rl_stocks.inference.canonical_tanh_policy import CanonicalTanhPolicy; import sys; a=CanonicalTanhPolicy.load(sys.argv[1]); assert a.predict([.5]+[0.]*20)[3]==.5; assert "torch" not in sys.modules and "numpy" not in sys.modules'
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0,sys.argv.pop(1)); " + code,
            str(__import__("pathlib").Path.cwd() / "src"),
            str(path),
        ],
        check=True,
        env=os.environ.copy(),
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.update(schema="canonical-task-actor-v1"),
        lambda b: b.update(input_size=22),
        lambda b: b.update(output_transform="tanh"),
        lambda b: b["layers"][0]["weights"][0].pop(),
        lambda b: b["layers"][1]["biases"].__setitem__(0, "7fc00000"),
    ],
)
def test_rejects_other_contracts_and_nonfinite_parameters(mutate):
    env = fixture()
    mutate(env["actor"])
    env["sha256"] = digest(env["actor"])
    with pytest.raises(ValueError):
        CanonicalTanhPolicy(env)


def test_export_preserves_rng_and_shared_reference_output_without_learning(tmp_path):
    x = np.random.default_rng(4).normal(size=(3, 21)).astype(np.float32)
    actors = []
    for algo in ("ppo", "trpo"):
        model = create(algo, ContractEnv())
        source = tmp_path / (algo + ".zip")
        model.save(source)
        raw = source.read_bytes()
        python = random.getstate()
        numpy = np.random.get_state()
        rng = torch.get_rng_state().clone()
        output = export_policy(source, tmp_path / (algo + ".json"), algorithm=algo)
        assert source.read_bytes() == raw and python == random.getstate()
        assert np.array_equal(numpy[1], np.random.get_state()[1]) and torch.equal(
            rng, torch.get_rng_state()
        )
        actor = CanonicalTanhPolicy.load(
            output, expected_source=hashlib.sha256(raw).hexdigest(), expected_algorithm=algo
        )
        actors.append([actor.predict(row.tolist()) for row in x])
        assert model.num_timesteps == 0 and not model.policy.optimizer.state
        with pytest.raises(ValueError, match="algorithm metadata"):
            export_policy(
                source,
                tmp_path / (algo + "-wrong.json"),
                algorithm="trpo" if algo == "ppo" else "ppo",
            )
        with pytest.raises(FileExistsError):
            export_policy(source, output, algorithm=algo)
        with pytest.raises(ValueError):
            CanonicalTanhPolicy.load(output, expected_source="b" * 64)
        with pytest.raises(ValueError):
            CanonicalTanhPolicy.load(output, expected_algorithm="sac")
    assert actors[0] == actors[1]
    with pytest.raises(ValueError):
        export_policy(tmp_path / "missing", tmp_path / "bad", algorithm="sac")

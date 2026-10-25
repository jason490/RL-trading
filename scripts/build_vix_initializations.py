"""Pin zero-column recurrent embeddings and original sampling RNG; never read datasets."""

import argparse
import hashlib
import json
from pathlib import Path

from gymnasium import Env, spaces
import numpy as np
import torch

from rl_stocks.rl.registry.factory import create
from rl_stocks.training.shared_initialization import read_initialization, write_initialization
from rl_stocks.training.vix_initialization import (
    map_initial_tensors,
    source_initialization,
    validate_mapping,
    seeded_fingerprints,
)
from rl_stocks.training.vix_spec import parameters, preregistration, PREREGISTRATION_SHA256
from rl_stocks.core.provenance import repository_state

ROOT = Path(__file__).resolve().parents[1]


class ModelSpaces(Env):
    observation_space = spaces.Box(-np.inf, np.inf, (23,), dtype=np.float32)
    action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / "configs/initializations"):
        raise ValueError("portable tracked configs/initializations output required")
    if args.output.exists():
        raise ValueError("new immutable VIX initializer directory required")
    git_before = repository_state(ROOT)
    spec = preregistration()
    torch.set_num_threads(1)
    records = {}
    proofs = {}
    streams = {}
    protected = {}
    manifest = ROOT / spec["shared_initialization"]["source_manifest"]
    protected[str(manifest)] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    args.output.mkdir(parents=True)
    for seed in spec["shared_initialization"]["seeds"]:
        entry, original, rng = source_initialization(seed)
        source = ROOT / entry["file"]
        protected[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        if torch.__version__.split("+")[0] != entry["torch_version"]:
            raise ValueError("pinned original Torch initialization version required")
        model = create(
            "recurrent_ppo", ModelSpaces(), seed=seed, device="cpu", overrides=parameters()
        )
        proof = map_initial_tensors(model, original)
        if seeded_fingerprints() != seeded_fingerprints(seed):
            raise ValueError("native NumPy/Python seeded streams changed")
        torch.set_rng_state(torch.from_numpy(rng))
        target = args.output / f"vix23-seed{seed}.npz"
        saved = write_initialization(model, target)
        restored, restored_rng = read_initialization(target.read_bytes(), saved)
        validate_mapping(original, restored)
        if not np.array_equal(rng, restored_rng):
            raise ValueError("original CPU sampling RNG changed")
        record = dict(file=str(target.relative_to(ROOT)), seed=seed, smoke=False, **saved)
        records[f"legacy-screen-{seed}"] = entry
        for arm in ("masked", "full"):
            records[f"{arm}-screen-{seed}"] = record
        if seed == 11:
            for arm in ("legacy", "masked", "full"):
                records[arm + "-smoke-11"] = dict(records[arm + "-screen-11"], smoke=True)
        proofs[str(seed)] = proof
        streams[str(seed)] = seeded_fingerprints()
        if streams[str(seed)] != seeded_fingerprints(seed):
            raise ValueError("serialization changed seeded NumPy/Python streams")
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != v for p, v in protected.items()):
        raise ValueError("original initializer assets changed")
    implementation = [
        "scripts/build_vix_initializations.py",
        "src/rl_stocks/training/vix_initialization.py",
        "src/rl_stocks/training/vix_spec.py",
    ]
    result = dict(
        version=1,
        preregistration_sha256=PREREGISTRATION_SHA256,
        generated_without_fitting=True,
        no_dataset_or_provider_access=True,
        artifacts=records,
        mapping_proofs=proofs,
        seeded_numpy_python=streams,
        original_assets_unchanged=True,
        original_assets_sha256=protected,
        implementation_sha256={
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in implementation
        },
        source_git_before_generation=git_before,
        scope="Three zero-column23-input embeddings plus pinned legacy21-input assets; full/masked share exact tensors/empty optimizers, all phases share original full64 CPU RNG. NumPy/Python seeded states verified. No21-versus23 trajectory equivalence claim.",
    )
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Generated3 untrained zero-column VIX snapshots; original common tensors/RNG preserved. No data or optimizer steps."
    )


if __name__ == "__main__":
    main()

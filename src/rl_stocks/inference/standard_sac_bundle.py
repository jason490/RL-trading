"""Separate full21 frozen SAC control; exposure execution is explicit at load time."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from rl_stocks.core.contracts import ACTION_SCHEMA, canonical, digest
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.environment.observations.state import observation_spec
from rl_stocks.inference.bundle import checksum
from rl_stocks.inference.canonical_actor import _hex, _unique
from rl_stocks.inference.canonical_math import CONTRACT
from rl_stocks.inference.canonical_sac_policy import CanonicalSACPolicy
from rl_stocks.inference.tanh_bundle import checked_files, FILES, FrozenTanhPolicy
from rl_stocks.risk.exposure import execution_contract, validate_execution_contract, filter_exposure

SCHEMA = "canonical-standard-sac-bundle-v1"
FIELDS = {"schema", "arithmetic", "dataset_sha256", "feature_schema", "names", "normalizer",
          "normalizer_sha256", "data_manifest", "observation", "action_schema", "action_execution",
          "simulation", "actor_sha256", "dependency_lock_sha256", "checkpoint"}


def metadata(manifest, *, feature_schema, feed, expected_execution):
    from rl_stocks.training.trpo_smoke import protocol
    spec = protocol()
    if (set(manifest) != FIELDS or manifest["schema"] != SCHEMA
            or manifest["arithmetic"] != CONTRACT or manifest["dataset_sha256"] != spec["dataset"]
            or manifest["feature_schema"] != feature_schema or feature_schema != "daily-indicators-v1"
            or manifest["data_manifest"]["provenance"]["capabilities"]["feed"] != feed
            or manifest["data_manifest"]["included_splits"] != ["train", "validation"]
            or manifest["normalizer_sha256"] != spec["normalizer_sha256"]
            or digest(manifest["normalizer"]) != spec["normalizer_sha256"]
            or manifest["observation"] != observation_spec("execution_full")
            or manifest["action_schema"] != ACTION_SCHEMA
            or manifest["simulation"] != asdict(SimulationConfig())):
        raise ValueError("standard SAC frozen schema/preprocessing/feed mismatch")
    contract = validate_execution_contract(manifest["action_execution"])
    if (expected_execution not in ("unrestricted", "unit_magnitude")
            or contract != execution_contract("unrestricted", expected_execution)):
        raise ValueError("explicit standard SAC execution mode mismatch")
    checkpoint = manifest["checkpoint"]
    if (set(checkpoint) != {"phase", "transitions", "model_sha256", "acceptance_sha256"}
            or checkpoint["phase"] not in ("selected", "final")
            or type(checkpoint["transitions"]) is not int or checkpoint["transitions"] < 1):
        raise ValueError("explicit accepted checkpoint identity required")
    for key in ("model_sha256", "acceptance_sha256"):
        _hex(checkpoint[key], 64)
    normalizer = normalizer_from(manifest["normalizer"])
    if len(normalizer.names) != 7 or normalizer.names != tuple(manifest["names"]):
        raise ValueError("exact seven-field standard SAC preprocessing required")
    symbols = manifest["data_manifest"]["universe"]
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("explicit unique research universe required")
    return normalizer, frozenset(symbols)


def freeze(actor_path, dataset, output, *, checkpoint, evaluation_execution):
    from rl_stocks.training.trpo_smoke import validate_dataset
    validate_dataset(dataset)
    actor_path, output = Path(actor_path), Path(output)
    actor = CanonicalSACPolicy.load(actor_path, expected_source=checkpoint["model_sha256"])
    original = checksum(actor_path)
    lock = Path(__file__).resolve().parents[3] / "requirements.lock"
    manifest = {"schema": SCHEMA, "arithmetic": CONTRACT, "dataset_sha256": digest(dataset),
                "feature_schema": dataset["schema"], "names": deepcopy(dataset["names"]),
                "normalizer": deepcopy(dataset["normalizer"]),
                "normalizer_sha256": digest(dataset["normalizer"]),
                "data_manifest": deepcopy(dataset["manifest"]),
                "observation": observation_spec("execution_full"), "action_schema": ACTION_SCHEMA,
                "action_execution": execution_contract("unrestricted", evaluation_execution),
                "simulation": asdict(SimulationConfig()), "actor_sha256": original,
                "dependency_lock_sha256": checksum(lock), "checkpoint": deepcopy(checkpoint)}
    metadata(manifest, feature_schema=dataset["schema"],
             feed=dataset["manifest"]["provenance"]["capabilities"]["feed"],
             expected_execution=evaluation_execution)
    if actor.manifest["actor"]["source_model_sha256"] != checkpoint["model_sha256"]:
        raise ValueError("checkpoint does not bind exported actor")
    output.mkdir(parents=True, exist_ok=True)
    target = output / digest(manifest)
    with TemporaryDirectory(prefix=".standard-sac-", dir=output) as temporary:
        temporary = Path(temporary)
        shutil.copyfile(actor_path, temporary / "actor.json")
        shutil.copyfile(lock, temporary / "requirements.lock")
        (temporary / "manifest.json").write_bytes(canonical(manifest))
        (temporary / "checksums.json").write_bytes(canonical({n: checksum(temporary / n) for n in FILES}))
        if checksum(actor_path) != original:
            raise ValueError("standard SAC source actor changed during freezing")
        target.mkdir(exist_ok=False)
        try:
            for name in FILES | {"checksums.json"}:
                shutil.copyfile(temporary / name, target / name)
        except BaseException:
            shutil.rmtree(target)
            raise
    return target


class FrozenStandardSACPolicy:
    # Reuse the exact full21 input preparation/time checks; no Tanh actor loading.
    observation = FrozenTanhPolicy.observation
    predict_features = FrozenTanhPolicy.predict_features
    manifest = FrozenTanhPolicy.manifest
    normalizer = FrozenTanhPolicy.normalizer

    def __init__(self, path, *, feature_schema, feed, expected_execution, expected_bundle_id=None):
        path = Path(path)
        hashes = checked_files(path)
        manifest = json.loads((path / "manifest.json").read_text(), object_pairs_hook=_unique)
        if (path.name != digest(manifest)
                or expected_bundle_id is not None and path.name != expected_bundle_id
                or manifest["actor_sha256"] != hashes["actor.json"]
                or manifest["dependency_lock_sha256"] != hashes["requirements.lock"]):
            raise ValueError("standard SAC frozen bundle identity mismatch")
        self._normalizer, self._symbols = metadata(manifest, feature_schema=feature_schema,
                                                  feed=feed, expected_execution=expected_execution)
        self._actor = CanonicalSACPolicy.load(path / "actor.json",
                                              expected_source=manifest["checkpoint"]["model_sha256"])
        self._execution = expected_execution
        self._manifest = canonical(manifest)

    def raw_action(self, observation):
        return self._actor.predict(observation)

    def predict(self, observation):
        return tuple(float(v) for v in filter_exposure(self.raw_action(observation), self._execution))

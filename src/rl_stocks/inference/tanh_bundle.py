"""Immutable PPO/TRPO reference bundles with frozen full21 preprocessing."""

from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from rl_stocks.core.contracts import ACTION_SCHEMA, canonical, digest
from rl_stocks.data.generation.datasets import normalizer_from
from rl_stocks.data.validation.bars import timestamp
from rl_stocks.environment.execution.simulator import SimulationConfig
from rl_stocks.environment.observations.state import observation_spec
from rl_stocks.inference.bundle import checksum
from rl_stocks.inference.canonical_actor import _unique
from rl_stocks.inference.canonical_math import CONTRACT, input32
from rl_stocks.inference.canonical_tanh_policy import CanonicalTanhPolicy

SCHEMA = "canonical-tanh-bundle-v1"
FILES = {"manifest.json", "actor.json", "requirements.lock"}
FIELDS = {"schema", "arithmetic", "algorithm", "dataset_sha256", "model_sha256",
          "feature_schema", "names", "normalizer", "normalizer_sha256", "data_manifest",
          "observation", "action_schema", "exposure_execution", "simulation",
          "actor_sha256", "dependency_lock_sha256"}


def checked_files(path):
    if path.is_symlink() or not path.is_dir():
        raise ValueError("regular frozen bundle directory required")
    if {p.name for p in path.iterdir()} != FILES | {"checksums.json"}:
        raise ValueError("exact frozen bundle file set required")
    for p in path.iterdir():
        if p.is_symlink() or not p.is_file() or p.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("bounded regular frozen bundle files required")
    hashes = json.loads((path / "checksums.json").read_text(), object_pairs_hook=_unique)
    if set(hashes) != FILES or any(checksum(path / name) != hashes[name] for name in FILES):
        raise ValueError("frozen bundle checksum mismatch")
    return hashes


def metadata(manifest, *, feature_schema, feed):
    from rl_stocks.training.trpo_smoke import protocol
    spec = protocol()
    if (
        set(manifest) != FIELDS or manifest["schema"] != SCHEMA
        or manifest["arithmetic"] != CONTRACT or manifest["algorithm"] not in ("ppo", "trpo")
        or manifest["dataset_sha256"] != spec["dataset"]
        or manifest["feature_schema"] != feature_schema
        or manifest["feature_schema"] != "daily-indicators-v1"
        or manifest["data_manifest"]["provenance"]["capabilities"]["feed"] != feed
        or manifest["normalizer_sha256"] != spec["normalizer_sha256"]
        or digest(manifest["normalizer"]) != manifest["normalizer_sha256"]
        or manifest["data_manifest"]["included_splits"] != ["train", "validation"]
        or manifest["observation"] != observation_spec("execution_full")
        or manifest["action_schema"] != ACTION_SCHEMA
        or manifest["exposure_execution"] != "unrestricted"
        or manifest["simulation"] != asdict(SimulationConfig())
    ):
        raise ValueError("frozen preprocessing/schema/feed/execution mismatch")
    normalizer = normalizer_from(manifest["normalizer"])
    if len(normalizer.names) != 7 or tuple(manifest["names"]) != normalizer.names:
        raise ValueError("exact seven-field preprocessing required")
    symbols = manifest["data_manifest"]["universe"]
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("explicit unique research universe required")
    return normalizer, frozenset(symbols)


def freeze(actor_path, dataset, output, *, algorithm):
    """Package a trusted exported actor; preserve source bytes and dataset identity."""
    from rl_stocks.training.trpo_smoke import validate_dataset
    validate_dataset(dataset)
    actor_path, output = Path(actor_path), Path(output)
    actor = CanonicalTanhPolicy.load(actor_path, expected_algorithm=algorithm)
    original = checksum(actor_path)
    lock = Path(__file__).resolve().parents[3] / "requirements.lock"
    manifest = {
        "schema": SCHEMA, "arithmetic": CONTRACT, "algorithm": algorithm,
        "dataset_sha256": digest(dataset),
        "model_sha256": actor.manifest["actor"]["source_model_sha256"],
        "feature_schema": dataset["schema"], "names": deepcopy(dataset["names"]),
        "normalizer": deepcopy(dataset["normalizer"]),
        "normalizer_sha256": digest(dataset["normalizer"]),
        "data_manifest": deepcopy(dataset["manifest"]),
        "observation": observation_spec("execution_full"), "action_schema": ACTION_SCHEMA,
        "exposure_execution": "unrestricted", "simulation": asdict(SimulationConfig()),
        "actor_sha256": original, "dependency_lock_sha256": checksum(lock),
    }
    feed = dataset["manifest"]["provenance"]["capabilities"]["feed"]
    metadata(manifest, feature_schema=dataset["schema"], feed=feed)
    output.mkdir(parents=True, exist_ok=True)
    target = output / digest(manifest)
    with TemporaryDirectory(prefix=".tanh-bundle-", dir=output) as folder:
        folder = Path(folder)
        shutil.copyfile(actor_path, folder / "actor.json")
        shutil.copyfile(lock, folder / "requirements.lock")
        (folder / "manifest.json").write_bytes(canonical(manifest))
        (folder / "checksums.json").write_bytes(canonical(
            {name: checksum(folder / name) for name in FILES}))
        if checksum(actor_path) != original:
            raise ValueError("source actor changed during freezing")
        target.mkdir(exist_ok=False)
        try:
            for name in FILES | {"checksums.json"}:
                shutil.copyfile(folder / name, target / name)
        except BaseException:
            shutil.rmtree(target)
            raise
    return target


class FrozenTanhPolicy:
    def __init__(self, path, *, feature_schema, feed, expected_bundle_id=None):
        path = Path(path)
        hashes = checked_files(path)
        manifest = json.loads((path / "manifest.json").read_text(), object_pairs_hook=_unique)
        if (digest(manifest) != path.name
                or expected_bundle_id is not None and expected_bundle_id != path.name
                or manifest["actor_sha256"] != hashes["actor.json"]
                or manifest["dependency_lock_sha256"] != hashes["requirements.lock"]):
            raise ValueError("frozen bundle identity mismatch")
        self._normalizer, self._symbols = metadata(manifest, feature_schema=feature_schema, feed=feed)
        self._actor = CanonicalTanhPolicy.load(path / "actor.json",
                                               expected_source=manifest["model_sha256"],
                                               expected_algorithm=manifest["algorithm"])
        self._manifest = canonical(manifest)

    @property
    def manifest(self):
        return json.loads(self._manifest)

    @property
    def normalizer(self):
        return self._normalizer

    def observation(self, features, names, portfolio):
        if len(portfolio) != 14 or any(isinstance(v, bool) for v in list(features) + list(portfolio)):
            raise ValueError("seven raw features and fourteen numeric portfolio fields required")
        return tuple(input32(v) for v in self._normalizer.transform(features, names) + list(portfolio))

    def predict(self, observation):
        return self._actor.predict(observation)

    def predict_features(self, features, names, portfolio, *, symbol,
                         available_time, decision_time, now, max_age_seconds=86400):
        if symbol not in self._symbols:
            raise ValueError("unknown frozen research symbol")
        if (type(max_age_seconds) not in (int, float) or not math.isfinite(max_age_seconds)
                or max_age_seconds <= 0):
            raise ValueError("positive finite maximum age required")
        available, decision, current = map(timestamp, (available_time, decision_time, now))
        if not available <= decision <= current or (current - available).total_seconds() > max_age_seconds:
            raise ValueError("unavailable or stale frozen inputs")
        return self.predict(self.observation(features, names, portfolio))

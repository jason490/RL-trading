"""Full local acceptance and unchanged numerical imports before remote admission."""

import ast
import json
from pathlib import Path
import subprocess

from rl_stocks.evaluation.tanh_path_acceptance import verify_directory
from rl_stocks.evaluation.tanh_paths import protocol
from rl_stocks.inference.bundle import checksum

BASELINE_REVISION = "36077043f6d8adeb21f9af1ad1aab29318233fd0"
BASELINE_SOURCE_HASH = "b8731d05a77a2505eb3ac045cff1e6c7db8398dbc70e4dc3d00288bf9e3a327a"
ENTRY = "rl_stocks.evaluation.tanh_paths"


def numerical_imports(root, entry=ENTRY):
    """Include declared internal imports, including imports inside functions."""
    root = Path(root)
    pending, visited, files = [entry], set(), {}
    while pending:
        module = pending.pop()
        if module in visited or not module.startswith("rl_stocks"):
            continue
        visited.add(module)
        stem = root / "src" / module.replace(".", "/")
        path = stem.with_suffix(".py")
        if not path.exists():
            path = stem / "__init__.py"
        if not path.is_file() or path.is_symlink():
            continue
        relative = str(path.relative_to(root))
        files[relative] = checksum(path)
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                pending.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    base = ".".join(parts[:len(parts) - node.level + 1] + ([base] if base else []))
                pending.append(base)
                pending.extend(base + "." + a.name for a in node.names if a.name != "*")
    if not files:
        raise ValueError("numerical source import closure is empty")
    return dict(sorted(files.items()))


def reviewed_numerical_sources(root):
    root = Path(root)
    inventory = numerical_imports(root)
    import hashlib
    for name, current in inventory.items():
        original = subprocess.run(["git", "show", BASELINE_REVISION + ":" + name],
                                  cwd=root, capture_output=True, check=True, timeout=10).stdout
        if hashlib.sha256(original).hexdigest() != current:
            raise ValueError("numerical calculation source changed since local replay: " + name)
    return inventory


def admit_local(root, dataset):
    root = Path(root)
    local = root / "artifacts/trpo-full-path-local-v1"
    if not (local / "result.json").is_file():
        raise ValueError("complete local economic-path result required before allocation")
    provenance = json.loads((local / "provenance.json").read_text())
    if (provenance["source"] != {"revision": BASELINE_REVISION, "dirty": ""}
            or provenance["source_hash"] != BASELINE_SOURCE_HASH):
        raise ValueError("original local replay provenance required")
    result = verify_directory(local, dataset)
    spec = protocol()
    if len(result["cases"]) != len(spec["cases"]):
        raise ValueError("all frozen cases must pass local acceptance")
    inventory = reviewed_numerical_sources(root)
    return {"local_result": str(local / "result.json"),
            "local_result_sha256": checksum(local / "result.json"),
            "local_provenance_sha256": checksum(local / "provenance.json"),
            "numerical_sources": inventory,
            "cases": len(result["cases"]),
            "episodes": sum(len(c["episodes"]) for c in result["cases"]),
            "decisions": sum(c["decisions"] for c in result["cases"]),
            "training_performed": False, "allocation_admitted": False}


def bundle_assets(root):
    """Exactly four checked bundles and the accepted public source receipt."""
    from rl_stocks.inference.tanh_bundle import FrozenTanhPolicy, FILES
    root = Path(root).resolve()
    spec = protocol()
    assets = []
    for case in spec["cases"]:
        path = root / case["bundle"]
        if not path.resolve().is_relative_to(root):
            raise ValueError("owned frozen bundle required")
        frozen = FrozenTanhPolicy(path, feature_schema="daily-indicators-v1", feed="yahoo",
                                  expected_bundle_id=case["bundle_id"])
        if (checksum(path / "actor.json") != case["actor_sha256"]
                or frozen.manifest["model_sha256"] != case["source_model_sha256"]):
            raise ValueError("original learned actor assets required")
        for name in sorted(FILES | {"checksums.json"}):
            assets.append((path / name, f'tanh-bundles/{case["case"]}/{case["bundle_id"]}/{name}'))
    receipt_name = "results/trpo-smoke-2026-11-08/frozen-preprocessing.json"
    if checksum(root / receipt_name) != spec["frozen_receipt_sha256"]:
        raise ValueError("accepted frozen source receipt required")
    assets.append((root / receipt_name, receipt_name))
    return assets


def prepare_path_upload(root, dataset, output, source, admission):
    from rl_stocks.core.contracts import canonical
    from rl_stocks.runtime.colab.trpo_smoke import prepare_upload
    root, output = Path(root), Path(output)
    if admission["numerical_sources"] != reviewed_numerical_sources(root):
        raise ValueError("admitted numerical source inventory changed")
    assets = bundle_assets(root)
    inventory = output.parent / "compute-sources.json"
    inventory.write_bytes(canonical(admission["numerical_sources"]))
    assets.append((inventory, "compute-sources.json"))
    return prepare_upload(root, dataset, output, source, extra_files=assets)

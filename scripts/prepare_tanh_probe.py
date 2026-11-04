"""Prepare the fixed untrained128-point probe without provisioning or fitting."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from rl_stocks.core.contracts import canonical
from rl_stocks.core.files import write_once
from rl_stocks.inference.bundle import checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    record = json.loads(
        (root / "results/trpo-feasibility-2026-11-04/local-reference.json").read_text()
    )
    probe = root / record["probe_path"]
    actor = root / "artifacts/trpo-reference-engineering-v1/reference.json"
    if checksum(probe) != record["probe_sha256"] or checksum(actor) != record["reference_sha256"]:
        raise ValueError("retained untrained engineering source changed")
    saved = json.loads(probe.read_text())
    payload = {"actor": json.loads(actor.read_text()), "inputs_hex": saved["inputs_hex"]}
    stage = root / ".runtime/trpo-tanh-probe-inputs-v1.json"
    write_once(stage, canonical(payload))
    code = "import json,sys;sys.path.insert(0,sys.argv[1]);from rl_stocks.inference.canonical_tanh_probe import execute;print(json.dumps(execute(json.load(open(sys.argv[2]))),allow_nan=False))"
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root / "src"), str(stage)],
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    expected = json.loads(result.stdout)
    if (
        expected["output_hex"] != saved["reference_outputs_hex"]
        or expected["native_packages_imported"]
    ):
        raise ValueError("isolated host reference outputs changed")
    write_once(args.output, canonical({"inputs": payload, "expected": expected}))
    print("Prepared128-point stdlib-only probe; no fitting or allocation.")


if __name__ == "__main__":
    main()

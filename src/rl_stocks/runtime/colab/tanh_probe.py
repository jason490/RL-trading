"""One stdlib-only Colab reference probe using the original metered lifecycle."""

import argparse
import base64
import fcntl
import io
import json
from pathlib import Path
import zipfile

from rl_stocks.core.contracts import digest
from rl_stocks.inference.canonical_tanh_policy import CanonicalTanhPolicy
from rl_stocks.inference.canonical_tanh_probe import verify, vectors
from rl_stocks.runtime.colab.reference_probe import ReferenceProbeJob, FILES

SOURCE_FILES = FILES[:-1] + (
    "rl_stocks/inference/canonical_tanh_policy.py",
    "rl_stocks/inference/canonical_tanh_probe.py",
)


PROTOCOL_SHA256 = "4e50eaa4cf31396172e397ad312a78b78085f4e69b7f29c040779e82a253219b"


def probe_spec(root):
    value = json.loads((Path(root) / "configs/experiments/trpo_tanh_probe.json").read_text())
    if digest(value) != PROTOCOL_SHA256:
        raise ValueError("fixed tanh probe protocol required")
    return value


def worker_script(root, payload, remote):
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCE_FILES:
            archive.writestr(name, (Path(root) / "src" / name).read_bytes())
    encoded = base64.b64encode(source.getvalue()).decode()
    return f"""import base64,json,pathlib,subprocess,sys,tempfile
assert sys.version_info[:2]==(3,13), "Inspect actual default kernel before changing project"
with tempfile.TemporaryDirectory(prefix="tanh-reference-") as folder:
    folder=pathlib.Path(folder)
    archive=folder/"source.zip";archive.write_bytes(base64.b64decode({encoded!r}))
    payload={payload!r}
    inputs=folder/"inputs.json";inputs.write_text(json.dumps(payload))
    output=pathlib.Path({remote!r});assert not output.exists(), "fresh probe output required"
    program="import json,sys; sys.path.insert(0,sys.argv[1]); from rl_stocks.inference.canonical_tanh_probe import execute; result=execute(json.load(open(sys.argv[2]))); open(sys.argv[3],'w').write(json.dumps(result,sort_keys=True,allow_nan=False))"
    subprocess.run([sys.executable,"-I","-c",program,str(archive),str(inputs),str(output)],check=True,timeout=180)
print("Tanh reference probe exported")
"""


class TanhProbeJob(ReferenceProbeJob):
    kind = "canonical-tanh-reference-probe"

    def validate_input(self, payload, expected):
        spec = probe_spec(self.root)
        vectors(payload["inputs_hex"])
        if (
            payload["actor"]["sha256"] != spec["actor_sha256"]
            or payload["actor"]["actor"]["source_model_sha256"] != spec["source_model_sha256"]
            or digest(payload["inputs_hex"]) != spec["inputs_sha256"]
            or expected["output_sha256"] != spec["outputs_sha256"]
            or self.limit_units != spec["maximum_probe_units"]
            or self.deadline_seconds != spec["deadline_seconds"]
            or self.reserve_seconds != spec["teardown_reserve_seconds"]
            or self.execution_seconds != spec["execution_seconds"]
        ):
            raise ValueError("fixed untrained synthetic probe and limits required")
        actor = CanonicalTanhPolicy(payload["actor"], expected_algorithm="trpo")
        if (
            actor.manifest["sha256"] != expected["actor_sha256"]
            or digest(payload["inputs_hex"]) != expected["inputs_sha256"]
        ):
            raise ValueError("bound synthetic actor/input source required")
        verify(expected, expected)

    def build_worker(self, inputs, remote):
        return worker_script(self.root, inputs, remote)

    def verify_result(self, actual, expected):
        return verify(actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--cli-python", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    payload = json.loads(args.input.read_text())
    with (root / ".runtime/allocation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        TanhProbeJob(root, args.cli_python, "CPU").run_probe(
            payload["inputs"],
            payload["expected"],
            root / ".runtime/trpo-tanh-reference-probe-v1-claim.json",
        )


if __name__ == "__main__":
    main()

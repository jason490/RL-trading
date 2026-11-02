"""Prepare the fixed three saved actors and isolated host synthetic reference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from rl_stocks.core.contracts import canonical, digest
from rl_stocks.inference.canonical_actor import ROLES
from rl_stocks.inference.canonical_export import export_actor
from rl_stocks.inference.canonical_probe import verify
from rl_stocks.runtime.colab.reference_probe import worker_script

SOURCES = {
    'masked_shared': ('rl-stocks-cpu-bf3f86798a17', 'task-screen/frozen-selected',
                      '48f55e6b9a558b6d943f93414de5f4d51e7496b25fe0680d59776adb1aee6422'),
    'tagged_shared': ('rl-stocks-cpu-8421f33527ca', 'task-screen/frozen-selected',
                      '16b5ce4a0d3ebe83331ec0a8afa25731ba52888cc46f32a253664ab11bfc2a72'),
    'tagged_routed': ('rl-stocks-cpu-c4d7b85c99ed', 'task-smoke/frozen',
                      '733a027dc3e883c26d6227896538bfd7aca8471e0602626a8befebd02bc0ddad'),
}


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    actors, sources = {}, {}
    for role in ROLES:
        session, folder, identity = SOURCES[role]
        source = root / 'artifacts/colab' / session / 'retrieved/artifacts' / folder / identity
        manifest = json.loads((source / 'manifest.json').read_text())
        if digest(manifest) != identity:
            raise ValueError('fixed source bundle identity mismatch')
        target = export_actor(source / 'model.zip', output / f'{role}.json')
        actors[role] = json.loads(target.read_text())
        if actors[role]['actor']['source_model_sha256'] != manifest['model_sha256']:
            raise ValueError('fixed source model checksum mismatch')
        sources[role] = {'session': session, 'bundle_id': identity,
                         'source_model_sha256': manifest['model_sha256']}
    script = output / 'host-probe.py'
    script.write_text(worker_script(root, actors, str(output / 'host-reference.json')))
    subprocess.run([sys.executable, str(script)], check=True, timeout=240)
    expected = json.loads((output / 'host-reference.json').read_text())
    verify(expected, expected)
    (output / 'input.json').write_bytes(canonical({'actors': actors, 'expected': expected}))
    (output / 'source-identities.json').write_bytes(canonical(sources))
    return output / 'input.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='.runtime/canonical-reference-probe-input-v1')
    args = parser.parse_args()
    print(prepare(Path.cwd(), args.output))

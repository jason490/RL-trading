"""The CLI's zero exit status alone does not prove remote Python succeeded."""
from collections import Counter
import hashlib
import json
from pathlib import PurePosixPath
import tarfile

from rl_stocks.core.contracts import digest


def validate_archive(path, plan, dataset_id):
    with tarfile.open(path) as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
        if 'artifacts/failure.txt' in names:
            raise RuntimeError('remote training failed; see retrieved artifacts/failure.txt')
        def read(name):
            stream = archive.extractfile(name)
            if stream is None:
                raise RuntimeError('missing required remote artifact: '+name)
            return stream.read()
        try:
            completion = json.loads(read('artifacts/completion.json'))
            if (completion['plan'] != plan or completion['dataset'] != dataset_id
                    or completion['python'][:2] != [3, 12] or 'A100' not in completion['gpu']):
                raise RuntimeError('remote completion manifest mismatch')
            runs = [json.loads(read(name)) for name in sorted(names)
                    if name.startswith('artifacts/runs/') and name.endswith('/metadata.json')]
            expected = Counter((j['algorithm'], j['device'], j['smoke'], j['steps'], plan['seed'])
                               for j in plan['jobs'])
            actual = Counter()
            for run in runs:
                if (run['status'] != 'complete' or run['dataset'] != dataset_id
                        or run['actual_steps'] != run['requested_steps']):
                    raise RuntimeError('incomplete remote training run')
                actual[(run['algorithm'], run['device'], run['smoke'], run['actual_steps'], run['seed'])] += 1
                folder = 'artifacts/frozen/'+run['bundle_id']+'/'
                hashes = json.loads(read(folder+'checksums.json'))
                if set(hashes) != {name[len(folder):] for name in names
                                   if name.startswith(folder) and name != folder+'checksums.json'}:
                    raise RuntimeError('remote bundle file set mismatch')
                for name, checksum in hashes.items():
                    if PurePosixPath(name).name != name or hashlib.sha256(read(folder+name)).hexdigest() != checksum:
                        raise RuntimeError('remote bundle checksum mismatch')
                if digest(json.loads(read(folder+'manifest.json'))) != run['bundle_id']:
                    raise RuntimeError('remote bundle identity mismatch')
            if actual != expected or len(completion['completed']) != len(runs):
                raise RuntimeError('missing or unexpected remote training jobs')
            return runs
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError('missing or malformed remote training artifacts') from exc

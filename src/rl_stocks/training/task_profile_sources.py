"""Require current compute dependencies to match the measured profile revision."""
import hashlib
from pathlib import Path
import re
import subprocess


SCOPES = ('src/rl_stocks/environment', 'src/rl_stocks/rl/sac',
          'src/rl_stocks/rl/registry/factory.py', 'src/rl_stocks/features',
          'src/rl_stocks/data/generation', 'src/rl_stocks/data/validation',
          'src/rl_stocks/data/universe', 'src/rl_stocks/core', 'src/rl_stocks/risk',
          'src/rl_stocks/evaluation', 'src/rl_stocks/training/task_environment.py',
          'src/rl_stocks/training/task_conditioning.py',
          'src/rl_stocks/training/task_optimizer.py', 'requirements.lock')


def verify_profile_sources(root, revision):
    root = Path(root)
    if not isinstance(revision, str) or not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('exact measured profile revision required')
    result = subprocess.run(['git', 'ls-tree', '-r', revision, '--', *SCOPES],
                            cwd=root, capture_output=True, text=True, timeout=30, check=True)
    original = {}
    for line in result.stdout.splitlines():
        descriptor, name = line.split('\t', 1)
        mode, kind, blob = descriptor.split()
        if name.endswith('.py') or name == 'requirements.lock':
            if mode != '100644' or kind != 'blob':
                raise ValueError('ordinary profiled compute source files required')
            original[name] = blob
    current = set()
    for name in SCOPES:
        path = root / name
        if path.is_dir():
            current.update(str(p.relative_to(root)) for p in path.rglob('*.py'))
        elif path.exists():
            current.add(name)
    if not original or set(original) != current:
        raise ValueError('profiled compute source inventory changed')
    hashes = {}
    for name, blob in original.items():
        path = root / name
        if path.is_symlink():
            raise ValueError('profiled compute source cannot be a symlink')
        content = path.read_bytes()
        actual_blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if actual_blob != blob:
            raise ValueError('profiled compute source changed: ' + name)
        hashes[name] = hashlib.sha256(content).hexdigest()
    return hashes

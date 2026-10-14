"""Package-source identity when a usable Git checkout is unavailable."""
import hashlib
from pathlib import Path
from rl_stocks.core.contracts import digest


def source_identity():
    root=Path(__file__).resolve().parents[1]
    files={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
           for path in sorted(root.rglob('*.py'))}
    return digest(files)

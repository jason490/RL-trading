"""Package-source identity when a usable Git checkout is unavailable."""
import hashlib
import json
from pathlib import Path
import subprocess
from rl_stocks.core.contracts import digest


def source_identity():
    root=Path(__file__).resolve().parents[1]
    files={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
           for path in sorted(root.rglob('*.py'))}
    return digest(files)


def repository_state(root):
    """Use captured source provenance when the remote payload has no .git tree."""
    root = Path(root)
    captured = root / "source-git.json"
    if captured.exists():
        state = json.loads(captured.read_text())
        if set(state) != {"revision", "dirty"}:
            raise ValueError("invalid captured Git provenance")
        if any(value is not None and not isinstance(value, str) for value in state.values()):
            raise ValueError("invalid captured Git provenance")
        return state
    def command(args):
        try:
            result = subprocess.run(["git", *args], cwd=root, capture_output=True,
                                    text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    return {"revision": command(["rev-parse", "HEAD"]),
            "dirty": command(["status", "--porcelain"])}

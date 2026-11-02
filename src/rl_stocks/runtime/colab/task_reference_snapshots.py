"""Validate a completed owned snapshot before promoting its unique local name."""

import gzip
from pathlib import Path, PurePosixPath
import tarfile

from rl_stocks.training.task_reference_runtime import checksum


def retain(candidate, target, previous):
    candidate, target = Path(candidate), Path(target)
    if target.exists():
        raise ValueError("retained checkpoint snapshot cannot be overwritten")
    total = 0
    with gzip.open(candidate, "rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            total += len(chunk)
            if total > 1024 * 1024 * 1024:
                raise ValueError("bounded checkpoint snapshot required")
    with tarfile.open(candidate) as stream:
        members = stream.getmembers()
        if len(members) > 64 or not members:
            raise ValueError("bounded nonempty checkpoint archive required")
        names = set()
        for member in members:
            name = PurePosixPath(member.name)
            if (
                member.name in names
                or name.is_absolute()
                or ".." in name.parts
                or "\\" in member.name
                or not name.parts
                or name.parts[0] != "artifacts"
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("ordinary owned checkpoint members required")
            names.add(member.name)
    sha = checksum(candidate)
    if sha in {r["sha256"] for r in previous}:
        candidate.unlink()  # Exact newly downloaded duplicate; prior bytes stay intact.
        return None
    size = candidate.stat().st_size
    candidate.rename(target)
    return {"path": str(target), "sha256": sha, "bytes": size}

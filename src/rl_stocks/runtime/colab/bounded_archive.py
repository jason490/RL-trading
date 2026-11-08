"""Validate an owned artifact container completely before extracting any member."""

from pathlib import Path, PurePosixPath
import tarfile


def extract_artifacts(path, destination, *, maximum_bytes, maximum_members):
    path, destination = Path(path), Path(destination)
    if (type(maximum_bytes) is not int or maximum_bytes < 1
            or type(maximum_members) is not int or maximum_members < 1
            or path.is_symlink() or not path.is_file() or destination.exists()
            or path.stat().st_size > maximum_bytes):
        raise ValueError("fresh extraction and bounded owned transport required")
    with tarfile.open(path) as archive:
        members, seen, total = [], set(), 0
        for member in archive:
            total += member.size
            if len(members) >= maximum_members or total > maximum_bytes:
                raise ValueError("artifact container count/expanded size exceeded")
            name = PurePosixPath(member.name)
            if (not (member.isfile() or member.isdir()) or member.size < 0
                    or name.is_absolute() or ".." in name.parts or not name.parts
                    or name.parts[0] != "artifacts" or str(name) in seen):
                raise ValueError("invalid bounded regular artifact container")
            seen.add(str(name))
            members.append(member)
        if not members:
            raise ValueError("empty artifact container")
        if "artifacts/failure.txt" in seen:
            raise RuntimeError("remote worker failed; retain archive for diagnosis")
        destination.mkdir(parents=True)
        archive.extractall(destination, filter="data")
    return destination / "artifacts"

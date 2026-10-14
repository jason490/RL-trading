"""Atomic, content-preserving writes for versioned research artifacts."""
import os
from pathlib import Path
import uuid


def write_once(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError('immutable artifact collision/corruption')
    finally:
        temporary.unlink(missing_ok=True)

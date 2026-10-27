"""Publish a complete owned artifact tree as the controller's checkpoint archive."""
import os
from pathlib import Path
import tarfile


def export_artifacts(artifact_root):
    destination = os.environ.get('RL_CHECKPOINT_ARCHIVE')
    if not destination:
        return
    root = Path(artifact_root).resolve()
    target = Path(destination).resolve()
    if root.name != 'artifacts' or not root.is_dir() or target.is_relative_to(root):
        raise ValueError('owned artifacts root and external checkpoint destination required')
    temporary = target.with_name(target.name+'.tmp')
    try:
        with tarfile.open(temporary, 'w:gz') as archive:
            archive.add(root, arcname='artifacts')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)

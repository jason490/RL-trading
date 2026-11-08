"""Exclude generated Python cache files from research source uploads."""

from pathlib import PurePosixPath


def source_member(member):
    path = PurePosixPath(member.name)
    if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
        return None
    return member

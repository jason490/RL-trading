"""Immutable ridge plus additive residual trees; no sklearn or pickle required."""

from dataclasses import dataclass
from pathlib import Path
import zipfile
from types import MappingProxyType

import numpy as np

from rl_stocks.inference.forest_arrays import FIELDS, predict_forest, validate_tree


@dataclass(frozen=True)
class ResidualBoosting:
    ridge: np.ndarray
    trees: tuple


def _validated(ridge, trees):
    ridge = np.array(ridge, dtype=np.float64, copy=True)
    if ridge.shape != (11,) or not np.isfinite(ridge).all():
        raise ValueError("finite ten-feature ridge plus intercept required")
    if len(trees) not in (2, 256):
        raise ValueError("fixed prefix/full boosting tree count required")
    result = []
    for tree in trees:
        t = validate_tree(tree)
        # The fixed depth-two learner has at most seven nodes. Check actual
        # topology too: seven nodes can otherwise form a deeper chain.
        if len(t["feature"]) > 7:
            raise ValueError("depth-two residual tree required")
        stack = [(0, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > 2:
                raise ValueError("depth-two residual tree required")
            if t["children_left"][node] != -1:
                stack.extend(
                    [
                        (int(t["children_left"][node]), depth + 1),
                        (int(t["children_right"][node]), depth + 1),
                    ]
                )
        t = {name: np.frombuffer(value.tobytes(), dtype=value.dtype) for name, value in t.items()}
        result.append(MappingProxyType(t))
    ridge = np.frombuffer(ridge.tobytes(), dtype=np.float64)
    return ResidualBoosting(ridge, tuple(result))


def save_boosting(path, ridge, trees):
    model = _validated(ridge, trees)
    arrays = {"ridge": model.ridge, "format_version": np.array([1], dtype=np.int64)}
    arrays.update(
        {f"{i:03d}_{name}": tree[name] for i, tree in enumerate(model.trees) for name in FIELDS}
    )
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_boosting(path, *, prefix=False):
    if type(prefix) is not bool:
        raise ValueError("explicit prefix mode required")
    count = 2 if prefix else 256
    expected = {"ridge", "format_version"} | {
        f"{i:03d}_{name}" for i in range(count) for name in FIELDS
    }
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if (
            len(infos) != len(expected)
            or len({i.filename for i in infos}) != len(infos)
            or {i.filename for i in infos} != {name + ".npy" for name in expected}
            or any(i.file_size > 4096 for i in infos)
        ):
            raise ValueError("bounded boosting archive members required")
    with np.load(path, allow_pickle=False) as source:
        version = source["format_version"]
        if version.dtype.kind not in "iu" or version.shape != (1,) or version[0] != 1:
            raise ValueError("unsupported boosting format")
        return _validated(
            source["ridge"],
            [{name: source[f"{i:03d}_{name}"] for name in FIELDS} for i in range(count)],
        )


def predict_boosting(model, features):
    raw = np.asarray(features, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 10 or not np.isfinite(raw).all():
        raise ValueError("finite ten-field boosting input required")
    # Native sklearn trees compare float32 inputs, while the unchanged ridge
    # uses float64. Keep the two computations separate before the final sum.
    residual = np.zeros(len(raw), dtype=np.float64)
    for tree in model.trees:
        residual += 0.03 * predict_forest([tree], raw)
    result = np.column_stack((np.ones(len(raw)), raw)) @ model.ridge + residual
    if not np.isfinite(result).all():
        raise ValueError("nonfinite boosted prediction")
    return result

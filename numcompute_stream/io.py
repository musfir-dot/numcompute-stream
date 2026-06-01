from __future__ import annotations

import csv

import numpy as np

__all__ = ["load_csv", "iter_chunks", "train_test_split"]

_MISSING = {"", "na", "nan", "n/a", "null", "none", "?"}


def load_csv(path, *, target: str | int | None = -1, header: bool = True,
             delimiter: str = ","):

    with open(path, newline="") as fh:
        rows = list(csv.reader(fh, delimiter=delimiter))
    if not rows:
        raise ValueError("CSV file is empty.")

    if header:
        names = rows[0]
        data = rows[1:]
    else:
        names = [f"col{i}" for i in range(len(rows[0]))]
        data = rows

    n_cols = len(names)
    columns = [[row[j] if j < len(row) else "" for row in data] for j in range(n_cols)]

    encoders: dict[int, list] = {}
    numeric_cols = []
    for j, col in enumerate(columns):
        numeric_cols.append(_encode_column(col, j, encoders))
    matrix = np.column_stack(numeric_cols) if numeric_cols else np.empty((len(data), 0))

    target_idx = _resolve_target(target, names)
    if target_idx is None:
        return {"X": matrix, "y": None, "feature_names": names, "encoders": encoders}

    y = matrix[:, target_idx]
    X = np.delete(matrix, target_idx, axis=1)
    feat_names = [n for k, n in enumerate(names) if k != target_idx]
    return {"X": X, "y": y, "feature_names": feat_names, "encoders": encoders}


def _encode_column(col, j, encoders) -> np.ndarray:

    parsed = np.empty(len(col), dtype=float)
    is_numeric = True
    for value in col:
        if value.strip().lower() in _MISSING:
            continue
        try:
            float(value)
        except ValueError:
            is_numeric = False
            break
    if is_numeric:
        for i, value in enumerate(col):
            parsed[i] = np.nan if value.strip().lower() in _MISSING else float(value)
        return parsed

    cats = sorted({v.strip() for v in col if v.strip().lower() not in _MISSING})
    mapping = {c: k for k, c in enumerate(cats)}
    encoders[j] = cats
    for i, value in enumerate(col):
        v = value.strip()
        parsed[i] = mapping.get(v, np.nan) if v.lower() not in _MISSING else np.nan
    return parsed


def _resolve_target(target, names):
    if target is None:
        return None
    if isinstance(target, str):
        if target not in names:
            raise ValueError(f"Target column {target!r} not found in {names}.")
        return names.index(target)
    idx = int(target)
    return idx % len(names)


def iter_chunks(X, y=None, *, n_chunks: int = 10, shuffle: bool = False,
                random_state: int | None = None):

    X = np.asarray(X)
    idx = np.arange(X.shape[0])
    if shuffle:
        np.random.default_rng(random_state).shuffle(idx)
    for part in np.array_split(idx, n_chunks):
        yield X[part], (None if y is None else np.asarray(y)[part])


def train_test_split(X, y, *, test_size: float = 0.25, random_state: int | None = None):
    X, y = np.asarray(X), np.asarray(y)
    n = X.shape[0]
    idx = np.arange(n)
    np.random.default_rng(random_state).shuffle(idx)
    cut = int(n * (1 - test_size))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]

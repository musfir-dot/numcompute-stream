from __future__ import annotations

import time

import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier


def _gini_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)


def naive_best_split(X: np.ndarray, y: np.ndarray):

    n_samples, n_features = X.shape
    classes = np.unique(y)
    parent_counts = np.array([(y == c).sum() for c in classes], dtype=float)
    parent_imp = _gini_from_counts(parent_counts)
    total = float(n_samples)

    best_gain, best_f, best_t = 0.0, -1, 0.0
    for f in range(n_features):
        col = X[:, f]
        order = np.argsort(col)
        xs, ys = col[order], y[order]
        for i in range(n_samples - 1):
            if xs[i] == xs[i + 1]:
                continue
            left_y, right_y = ys[: i + 1], ys[i + 1:]
            lc = np.array([(left_y == c).sum() for c in classes], dtype=float)
            rc = np.array([(right_y == c).sum() for c in classes], dtype=float)
            lw, rw = lc.sum(), rc.sum()
            weighted = (lw * _gini_from_counts(lc) + rw * _gini_from_counts(rc)) / total
            gain = parent_imp - weighted
            if gain > best_gain:
                best_gain, best_f = gain, f
                best_t = (xs[i] + xs[i + 1]) / 2.0
    return best_f, best_t


def vectorised_best_split(X: np.ndarray, y: np.ndarray):
    """The framework's own vectorised search, exposed for timing."""
    tree = DecisionTreeClassifier(criterion="gini")
    tree.classes_ = np.unique(y)
    w = np.ones(len(y))
    return tree._best_split(X, y, w)


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"{'n_samples':>10} | {'naive (s)':>10} | {'vector (s)':>11} | {'speed-up':>9}")
    print("-" * 50)
    for n in (500, 1000, 2000, 4000):
        X = rng.normal(size=(n, 8))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        t0 = time.perf_counter()
        naive_best_split(X, y)
        t_naive = time.perf_counter() - t0

        t0 = time.perf_counter()
        vectorised_best_split(X, y)
        t_vec = time.perf_counter() - t0

        speed = t_naive / t_vec if t_vec > 0 else float("inf")
        print(f"{n:>10} | {t_naive:>10.4f} | {t_vec:>11.4f} | {speed:>8.1f}x")


if __name__ == "__main__":
    main()

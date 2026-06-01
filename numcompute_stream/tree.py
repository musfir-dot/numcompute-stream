from __future__ import annotations

import numpy as np

from ._base import StreamEstimator, check_X_y, check_array

__all__ = ["DecisionTreeClassifier"]


class _Node:

    __slots__ = ("feature", "threshold", "left", "right", "proba", "is_leaf")

    def __init__(self):
        self.feature = -1
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.proba = None
        self.is_leaf = False


class DecisionTreeClassifier(StreamEstimator):


    def __init__(self, max_depth: int = 8, min_samples_split: int = 2,
                 max_features=None, criterion: str = "gini",
                 max_buffer: int = 10000, random_state: int | None = None):
        if criterion not in {"gini", "entropy"}:
            raise ValueError(f"Unknown criterion {criterion!r}.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.max_buffer = max_buffer
        self.random_state = random_state
        self.reset()

    def reset(self) -> "DecisionTreeClassifier":
        self.classes_ = None
        self.root_ = None
        self._Xbuf = None
        self._ybuf = None
        self._wbuf = None
        self._rng = np.random.default_rng(self.random_state)
        return self


    def partial_fit(self, X, y, classes=None, sample_weight=None) -> "DecisionTreeClassifier":
        """Append a labelled chunk to the buffer and re-induce the tree."""
        X, y = check_X_y(X, y, allow_nan=False)
        w = (np.ones(X.shape[0]) if sample_weight is None
             else np.asarray(sample_weight, dtype=float).ravel())
        if w.shape[0] != X.shape[0]:
            raise ValueError("sample_weight length must match number of samples.")

        if self._Xbuf is None:
            self._Xbuf, self._ybuf, self._wbuf = X, y, w
        else:
            if X.shape[1] != self._Xbuf.shape[1]:
                raise ValueError("Feature count changed between chunks.")
            self._Xbuf = np.vstack([self._Xbuf, X])
            self._ybuf = np.concatenate([self._ybuf, y])
            self._wbuf = np.concatenate([self._wbuf, w])

        if self._Xbuf.shape[0] > self.max_buffer:
            keep = slice(self._Xbuf.shape[0] - self.max_buffer, None)
            self._Xbuf, self._ybuf, self._wbuf = (
                self._Xbuf[keep], self._ybuf[keep], self._wbuf[keep])

        known = np.unique(self._ybuf)
        if classes is not None:
            known = np.unique(np.concatenate([known, np.asarray(classes).ravel()]))
        self.classes_ = known
        self.root_ = self._build(self._Xbuf, self._ybuf, self._wbuf, depth=0)
        return self

    def _n_features_to_consider(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(mf, float):
            return max(1, int(mf * n_features))
        return max(1, min(int(mf), n_features))

    def _class_proba(self, y, w) -> np.ndarray:

        proba = np.zeros(self.classes_.shape[0])
        for i, c in enumerate(self.classes_):
            proba[i] = w[y == c].sum()
        total = proba.sum()
        return proba / total if total > 0 else proba

    def _build(self, X, y, w, depth: int) -> _Node:
        node = _Node()
        node.proba = self._class_proba(y, w)

        if (depth >= self.max_depth or w.sum() < self.min_samples_split
                or np.unique(y).size <= 1):
            node.is_leaf = True
            return node

        feature, threshold = self._best_split(X, y, w)
        if feature < 0:
            node.is_leaf = True
            return node

        mask = X[:, feature] <= threshold
        if mask.all() or (~mask).all():
            node.is_leaf = True
            return node

        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[mask], y[mask], w[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], w[~mask], depth + 1)
        return node

    def _impurity_from_counts(self, counts: np.ndarray) -> np.ndarray:

        totals = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(totals > 0, counts / totals, 0.0)
        if self.criterion == "gini":
            return 1.0 - (p**2).sum(axis=1)
        safe_p = np.where(p > 0, p, 1.0)
        logp = np.log2(safe_p)
        return -(np.where(p > 0, p * logp, 0.0)).sum(axis=1)

    def _best_split(self, X, y, w):
        n_samples, n_features = X.shape
        y_enc = np.searchsorted(self.classes_, y)
        k = self.classes_.shape[0]

        feat_idx = np.arange(n_features)
        n_consider = self._n_features_to_consider(n_features)
        if n_consider < n_features:
            feat_idx = self._rng.choice(n_features, size=n_consider, replace=False)

        total_counts = np.zeros(k)
        np.add.at(total_counts, y_enc, w)
        parent_impurity = self._impurity_from_counts(total_counts[None, :])[0]
        total_w = w.sum()

        best_gain = 1e-12
        best_feature, best_threshold = -1, 0.0

        for f in feat_idx:
            col = X[:, f]
            order = np.argsort(col, kind="mergesort")
            xs, ys, ws = col[order], y_enc[order], w[order]

            onehot = np.zeros((n_samples, k))
            onehot[np.arange(n_samples), ys] = ws
            left_counts = np.cumsum(onehot, axis=0)        # (n, k)
            right_counts = total_counts[None, :] - left_counts

            left_w = left_counts.sum(axis=1)
            right_w = total_w - left_w

            valid = np.empty(n_samples, dtype=bool)
            valid[:-1] = xs[:-1] != xs[1:]
            valid[-1] = False
            valid &= (left_w > 0) & (right_w > 0)
            if not valid.any():
                continue

            li = self._impurity_from_counts(left_counts)
            ri = self._impurity_from_counts(right_counts)
            weighted = (left_w * li + right_w * ri) / total_w
            gain = parent_impurity - weighted
            gain[~valid] = -np.inf

            pos = int(np.argmax(gain))
            if gain[pos] > best_gain:
                best_gain = gain[pos]
                best_feature = int(f)
                best_threshold = (xs[pos] + xs[pos + 1]) / 2.0

        return best_feature, best_threshold


    def predict_proba(self, X) -> np.ndarray:
        X = check_array(X, allow_nan=False)
        if self.root_ is None:
            raise RuntimeError("Tree must be fitted before predict_proba.")
        out = np.zeros((X.shape[0], self.classes_.shape[0]))
        self._route(self.root_, X, np.arange(X.shape[0]), out)
        return out

    def _route(self, node: _Node, X, idx, out) -> None:
        if node.is_leaf or node.feature < 0:
            out[idx] = node.proba
            return
        go_left = X[idx, node.feature] <= node.threshold
        if go_left.any():
            self._route(node.left, X, idx[go_left], out)
        if (~go_left).any():
            self._route(node.right, X, idx[~go_left], out)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def depth(self) -> int:
        def _d(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_d(node.left), _d(node.right))
        return _d(self.root_) if self.root_ else 0


from __future__ import annotations

import numpy as np

from ._base import StreamEstimator, check_X_y, check_array
from .tree import DecisionTreeClassifier

__all__ = ["EnsembleClassifier", "RandomForestClassifier"]


class EnsembleClassifier(StreamEstimator):


    def __init__(self, n_estimators: int = 10, max_depth: int = 8,
                 max_features="sqrt", criterion: str = "gini",
                 bootstrap: bool = True, streaming_mode: str = "online_bagging",
                 max_buffer: int = 10000, random_state: int | None = None):
        if streaming_mode not in {"online_bagging", "rebuild"}:
            raise ValueError(f"Unknown streaming_mode {streaming_mode!r}.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.streaming_mode = streaming_mode
        self.max_buffer = max_buffer
        self.random_state = random_state
        self.reset()

    def reset(self) -> "EnsembleClassifier":
        self._rng = np.random.default_rng(self.random_state)
        self.estimators_ = [
            DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features,
                criterion=self.criterion,
                max_buffer=self.max_buffer,
                random_state=None if self.random_state is None else self.random_state + i,
            )
            for i in range(self.n_estimators)
        ]
        self.classes_ = None
        return self

    def _merge_classes(self, y, classes=None) -> np.ndarray:
        known = np.unique(y)
        if classes is not None:
            known = np.unique(np.concatenate([known, np.asarray(classes).ravel()]))
        if self.classes_ is not None:
            known = np.unique(np.concatenate([known, self.classes_]))
        return known

    def fit(self, X, y, sample_weight=None) -> "EnsembleClassifier":
        """Batch-fit each tree on an independent bootstrap sample."""
        X, y = check_X_y(X, y, allow_nan=False)
        self.reset()
        self.classes_ = self._merge_classes(y)
        n = X.shape[0]
        for tree in self.estimators_:
            if self.bootstrap:
                idx = self._rng.integers(0, n, size=n)
            else:
                idx = np.arange(n)
            tree.partial_fit(X[idx], y[idx], classes=self.classes_)
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None) -> "EnsembleClassifier":

        X, y = check_X_y(X, y, allow_nan=False)
        self.classes_ = self._merge_classes(y, classes)
        base_w = (np.ones(X.shape[0]) if sample_weight is None
                  else np.asarray(sample_weight, dtype=float).ravel())

        for tree in self.estimators_:
            if self.streaming_mode == "online_bagging":
                k = self._rng.poisson(1.0, size=X.shape[0]).astype(float)
                w = base_w * k
                if w.sum() <= 0:  # ensure the tree still sees something
                    w = base_w
            else:  # 'rebuild' -> resample the incoming chunk with replacement
                idx = self._rng.integers(0, X.shape[0], size=X.shape[0])
                tree.partial_fit(X[idx], y[idx], classes=self.classes_,
                                 sample_weight=base_w[idx])
                continue
            tree.partial_fit(X, y, classes=self.classes_, sample_weight=w)
        return self

    def predict_proba(self, X) -> np.ndarray:

        X = check_array(X, allow_nan=False)
        if self.classes_ is None:
            raise RuntimeError("Ensemble must be fitted before predict_proba.")
        agg = np.zeros((X.shape[0], self.classes_.shape[0]))
        n_used = 0
        for tree in self.estimators_:
            if tree.root_ is None:
                continue
            tree_proba = tree.predict_proba(X)
            cols = np.searchsorted(self.classes_, tree.classes_)
            agg[:, cols] += tree_proba
            n_used += 1
        if n_used:
            agg /= n_used
        return agg

    def predict(self, X) -> np.ndarray:

        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class RandomForestClassifier(EnsembleClassifier):


    def __init__(self, n_estimators: int = 20, max_depth: int = 10,
                 criterion: str = "gini", streaming_mode: str = "online_bagging",
                 max_buffer: int = 10000, random_state: int | None = None):
        super().__init__(
            n_estimators=n_estimators, max_depth=max_depth, max_features="sqrt",
            criterion=criterion, bootstrap=True, streaming_mode=streaming_mode,
            max_buffer=max_buffer, random_state=random_state,
        )

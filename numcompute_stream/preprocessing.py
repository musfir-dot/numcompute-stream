
from __future__ import annotations

import numpy as np

from ._base import StreamTransformer, check_array
from .stats import RunningMoments

__all__ = ["StandardScaler", "Imputer", "OneHotEncoder"]


class StandardScaler(StreamTransformer):


    def __init__(self, with_mean: bool = True, with_std: bool = True, eps: float = 1e-8):
        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = eps
        self.reset()

    def reset(self) -> "StandardScaler":
        self._moments = RunningMoments()
        return self

    def partial_fit(self, X, y=None) -> "StandardScaler":
        """Update running mean/variance from a feature chunk."""
        self._moments.update(check_array(X, allow_nan=True))
        return self

    def transform(self, X) -> np.ndarray:
        """Standardise ``X`` using statistics learned so far."""
        X = check_array(X, allow_nan=True)
        if self._moments.mean_ is None:
            raise RuntimeError("StandardScaler must see at least one chunk before transform.")
        out = X.astype(float, copy=True)
        if self.with_mean:
            out = out - self._moments.mean_
        if self.with_std:
            std = np.sqrt(self._moments.variance_)
            out = out / np.maximum(std, self.eps)
        return out

    @property
    def mean_(self):
        return self._moments.mean_

    @property
    def var_(self):
        return self._moments.variance_


class Imputer(StreamTransformer):


    def __init__(self, strategy: str = "mean", fill_value: float = 0.0):
        if strategy not in {"mean", "median", "constant"}:
            raise ValueError(f"Unknown strategy {strategy!r}.")
        self.strategy = strategy
        self.fill_value = fill_value
        self.reset()

    def reset(self) -> "Imputer":
        self._moments = RunningMoments()
        self._medians = None  
        return self

    def partial_fit(self, X, y=None) -> "Imputer":

        X = check_array(X, allow_nan=True)
        if self.strategy == "mean":
            self._moments.update(X)
        elif self.strategy == "median":
            from .stats import StreamingQuantile
            if self._medians is None:
                self._medians = [StreamingQuantile(0.5) for _ in range(X.shape[1])]
            for j, q in enumerate(self._medians):
                q.update(X[:, j])
        return self

    def _fill_values(self, n_features: int) -> np.ndarray:
        if self.strategy == "constant":
            return np.full(n_features, self.fill_value)
        if self.strategy == "mean":
            if self._moments.mean_ is None:
                return np.full(n_features, self.fill_value)
            return self._moments.mean_
        return np.array([q.value_ for q in self._medians])

    def transform(self, X) -> np.ndarray:

        X = check_array(X, allow_nan=True)
        fill = self._fill_values(X.shape[1])
        fill = np.where(np.isnan(fill), self.fill_value, fill)
        mask = np.isnan(X)
        out = X.copy()
        out[mask] = np.take(fill, np.where(mask)[1])
        return out


class OneHotEncoder(StreamTransformer):


    def __init__(self, columns=None):
        self.columns = None if columns is None else list(columns)
        self.reset()

    def reset(self) -> "OneHotEncoder":
        self.categories_: dict[int, list] = {}
        return self

    def _cols(self, n_features: int):
        return self.columns if self.columns is not None else list(range(n_features))

    def partial_fit(self, X, y=None) -> "OneHotEncoder":

        X = check_array(X, allow_nan=True)
        for c in self._cols(X.shape[1]):
            seen = self.categories_.setdefault(c, [])
            col = X[:, c]
            for v in np.unique(col[~np.isnan(col)]):
                if v not in seen:
                    seen.append(v)
        return self

    def output_width(self) -> int:

        return sum(len(v) for v in self.categories_.values())

    def transform(self, X) -> np.ndarray:

        X = check_array(X, allow_nan=True)
        blocks = []
        for c in self._cols(X.shape[1]):
            cats = self.categories_.get(c, [])
            block = np.zeros((X.shape[0], len(cats)))
            col = X[:, c]
            for j, cat in enumerate(cats):
                block[:, j] = (col == cat).astype(float)
            blocks.append(block)
        if not blocks:
            return np.empty((X.shape[0], 0))
        return np.hstack(blocks)

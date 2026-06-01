
from __future__ import annotations

import numpy as np

from ._base import BaseComponent, check_array

__all__ = [
    "RunningMoments",
    "StreamingQuantile",
    "StreamingHistogram",
    "update_stats",
]


class RunningMoments(BaseComponent):

    def __init__(self, n_features: int | None = None):
        self.n_features = n_features
        self.reset()

    def reset(self) -> "RunningMoments":
        """Clear all accumulated statistics."""
        n = self.n_features
        self.count_ = np.zeros(n) if n else None
        self.mean_ = np.zeros(n) if n else None
        self._m2 = np.zeros(n) if n else None
        return self

    def _ensure_init(self, n_features: int) -> None:
        if self.count_ is None:
            self.n_features = n_features
            self.count_ = np.zeros(n_features)
            self.mean_ = np.zeros(n_features)
            self._m2 = np.zeros(n_features)
        elif n_features != self.n_features:
            raise ValueError(
                f"Chunk has {n_features} features but estimator was "
                f"initialised with {self.n_features}."
            )

    def update(self, X_chunk) -> "RunningMoments":
        """Fold a chunk into the running moments using the parallel update.

        Implements the chunk form of Welford's algorithm (Chan et al.), which
        combines an existing aggregate with the aggregate of a new batch in a
        numerically stable way.
        """
        X = check_array(X_chunk, allow_nan=True, name="X_chunk")
        self._ensure_init(X.shape[1])

        mask = ~np.isnan(X)
        batch_count = mask.sum(axis=0).astype(float)
        safe = batch_count > 0
        if not safe.any():
            return self

        batch_mean = np.zeros(self.n_features)
        batch_m2 = np.zeros(self.n_features)
        filled = np.where(mask, X, 0.0)
        batch_mean[safe] = filled[:, safe].sum(axis=0) / batch_count[safe]
        dev = np.where(mask, X - batch_mean, 0.0)
        batch_m2[safe] = (dev[:, safe] ** 2).sum(axis=0)

        delta = batch_mean - self.mean_
        total = self.count_ + batch_count
        combined = np.where(total > 0, total, 1.0)

        self.mean_ = np.where(
            safe, self.mean_ + delta * (batch_count / combined), self.mean_
        )
        self._m2 = np.where(
            safe,
            self._m2 + batch_m2 + delta**2 * (self.count_ * batch_count / combined),
            self._m2,
        )
        self.count_ = total
        return self

    @property
    def variance_(self) -> np.ndarray:
        return np.where(self.count_ > 0, self._m2 / np.maximum(self.count_, 1), 0.0)

    @property
    def sample_variance_(self) -> np.ndarray:
        denom = np.maximum(self.count_ - 1, 1)
        return np.where(self.count_ > 1, self._m2 / denom, 0.0)

    @property
    def std_(self) -> np.ndarray:
        return np.sqrt(self.variance_)


class StreamingQuantile(BaseComponent):


    def __init__(self, q: float = 0.5):
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}.")
        self.q = q
        self.reset()

    def reset(self) -> "StreamingQuantile":
        self._n = 0
        self._q = np.zeros(5)         
        self._npos = np.arange(1, 6)  

        self._np_desired = np.array([1, 1 + 2 * self.q, 1 + 4 * self.q, 3 + 2 * self.q, 5])
        self._dn = np.array([0, self.q / 2, self.q, (1 + self.q) / 2, 1])
        return self

    def _parabolic(self, i: int, d: int) -> float:
        q, n = self._q, self._npos
        return q[i] + d / (n[i + 1] - n[i - 1]) * (
            (n[i] - n[i - 1] + d) * (q[i + 1] - q[i]) / (n[i + 1] - n[i])
            + (n[i + 1] - n[i] - d) * (q[i] - q[i - 1]) / (n[i] - n[i - 1])
        )

    def _linear(self, i: int, d: int) -> float:
        q, n = self._q, self._npos
        return q[i] + d * (q[i + d] - q[i]) / (n[i + d] - n[i])

    def update(self, X_chunk) -> "StreamingQuantile":
        """Stream a 1-D chunk of observations into the estimator."""
        values = np.asarray(X_chunk, dtype=float).ravel()
        values = values[~np.isnan(values)]
        for x in values:
            self._observe(float(x))
        return self

    def _observe(self, x: float) -> None:
        if self._n < 5:
            self._q[self._n] = x
            self._n += 1
            if self._n == 5:
                self._q.sort()
            return


        if x < self._q[0]:
            self._q[0] = x
            k = 0
        elif x >= self._q[4]:
            self._q[4] = x
            k = 3
        else:
            k = np.searchsorted(self._q, x, side="right") - 1
            k = int(np.clip(k, 0, 3))

        self._npos[k + 1:] += 1
        self._np_desired = self._np_desired + self._dn
        self._n += 1

        for i in range(1, 4):
            d = self._np_desired[i] - self._npos[i]
            left = self._npos[i] - self._npos[i - 1]
            right = self._npos[i + 1] - self._npos[i]
            if (d >= 1 and right > 1) or (d <= -1 and left > 1):
                step = int(np.sign(d))
                candidate = self._parabolic(i, step)
                if not (self._q[i - 1] < candidate < self._q[i + 1]):
                    candidate = self._linear(i, step)
                self._q[i] = candidate
                self._npos[i] += step

    @property
    def value_(self) -> float:
        if self._n == 0:
            return float("nan")
        if self._n < 5:
            return float(np.median(self._q[: self._n]))
        return float(self._q[2])


class StreamingHistogram(BaseComponent):


    def __init__(self, bins: int = 10, value_range=(0.0, 1.0), window: int | None = None):
        if bins < 1:
            raise ValueError("bins must be >= 1.")
        self.bins = bins
        self.value_range = value_range
        self.window = window
        self.reset()

    def reset(self) -> "StreamingHistogram":
        self.counts_ = np.zeros(self.bins, dtype=float)
        self.edges_ = np.linspace(self.value_range[0], self.value_range[1], self.bins + 1)
        self._buffer: list[float] = []
        return self

    def update(self, X_chunk) -> "StreamingHistogram":
        """Add a chunk of observations to the histogram."""
        values = np.asarray(X_chunk, dtype=float).ravel()
        values = values[~np.isnan(values)]
        clipped = np.clip(values, self.edges_[0], self.edges_[-1])
        idx = np.clip(np.digitize(clipped, self.edges_) - 1, 0, self.bins - 1)
        np.add.at(self.counts_, idx, 1.0)

        if self.window is not None:
            self._buffer.extend(clipped.tolist())
            while len(self._buffer) > self.window:
                old = self._buffer.pop(0)
                old_idx = int(np.clip(np.digitize(old, self.edges_) - 1, 0, self.bins - 1))
                self.counts_[old_idx] = max(0.0, self.counts_[old_idx] - 1.0)
        return self

    @property
    def density_(self) -> np.ndarray:
        total = self.counts_.sum()
        width = self.edges_[1] - self.edges_[0]
        if total == 0 or width == 0:
            return np.zeros_like(self.counts_)
        return self.counts_ / (total * width)


def update_stats(state: dict | None, X_chunk, *, bins: int = 10,
                 value_range=(0.0, 1.0)) -> dict:
    X = check_array(X_chunk, allow_nan=True, name="X_chunk")
    if state is None:
        state = {
            "moments": RunningMoments(),
            "histograms": [StreamingHistogram(bins, value_range) for _ in range(X.shape[1])],
        }
    state["moments"].update(X)
    for j, hist in enumerate(state["histograms"]):
        hist.update(X[:, j])
    return state

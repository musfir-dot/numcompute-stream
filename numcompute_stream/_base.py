
from __future__ import annotations

import numpy as np

__all__ = [
    "BaseComponent",
    "StreamTransformer",
    "StreamEstimator",
    "check_array",
    "check_X_y",
]


def check_array(X, *, allow_nan: bool = True, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be 1-D or 2-D, got {arr.ndim} dimensions "
            f"with shape {arr.shape}."
        )
    if arr.size == 0:
        raise ValueError(f"{name} is empty (shape {arr.shape}).")
    if not allow_nan and np.isnan(arr).any():
        raise ValueError(
            f"{name} contains NaN values but this component does not accept "
            f"them. Run an Imputer first."
        )
    return np.ascontiguousarray(arr)


def check_X_y(X, y, *, allow_nan: bool = True):

    X = check_array(X, allow_nan=allow_nan, name="X")
    y = np.asarray(y).ravel()
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} samples but y has {y.shape[0]}; "
            f"the first dimensions must match."
        )
    return X, y


class BaseComponent:

    def get_params(self) -> dict:
        params = {}
        for key, value in vars(self).items():
            if not key.endswith("_") and not key.startswith("_"):
                params[key] = value
        return params

    def __repr__(self) -> str:  
        items = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({items})"


class StreamTransformer(BaseComponent):

    def partial_fit(self, X, y=None) -> "StreamTransformer":
        raise NotImplementedError

    def transform(self, X) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X, y=None) -> "StreamTransformer":
        return self.reset().partial_fit(X, y)

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def reset(self) -> "StreamTransformer":
        return self


class StreamEstimator(BaseComponent):


    def partial_fit(self, X, y, classes=None, sample_weight=None) -> "StreamEstimator":
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X, y, sample_weight=None) -> "StreamEstimator":
        return self.reset().partial_fit(X, y, sample_weight=sample_weight)

    def reset(self) -> "StreamEstimator":
        return self

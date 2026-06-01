
from __future__ import annotations

import numpy as np

from ._base import StreamEstimator, StreamTransformer, check_array

__all__ = ["Pipeline"]


class Pipeline(StreamEstimator):


    def __init__(self, steps):
        names = [name for name, _ in steps]
        if len(names) != len(set(names)):
            raise ValueError("Pipeline step names must be unique.")
        self.steps = list(steps)
        self.reset()

    def reset(self) -> "Pipeline":
        for _, step in self.steps:
            if hasattr(step, "reset"):
                step.reset()
        return self

    @property
    def named_steps(self) -> dict:

        return dict(self.steps)

    @property
    def _final(self):
        return self.steps[-1][1]

    @property
    def classes_(self):
        return getattr(self._final, "classes_", None)

    def _transform_chunk(self, X, y, *, update: bool):

        Xt = check_array(X, allow_nan=True)
        for _, step in self.steps[:-1]:
            if update:
                step.partial_fit(Xt, y)
            Xt = step.transform(Xt)
        return Xt

    def partial_fit(self, X, y=None, classes=None, sample_weight=None) -> "Pipeline":

        Xt = self._transform_chunk(X, y, update=True)
        final = self._final
        if isinstance(final, StreamEstimator):
            final.partial_fit(Xt, y, classes=classes, sample_weight=sample_weight)
        else:
            final.partial_fit(Xt, y)
        return self

    def transform(self, X) -> np.ndarray:

        Xt = self._transform_chunk(X, None, update=False)
        final = self._final
        if isinstance(final, StreamTransformer):
            Xt = final.transform(Xt)
        return Xt

    def predict(self, X) -> np.ndarray:

        Xt = self._transform_chunk(X, None, update=False)
        return self._final.predict(Xt)

    def predict_proba(self, X) -> np.ndarray:

        Xt = self._transform_chunk(X, None, update=False)
        return self._final.predict_proba(Xt)



from __future__ import annotations

import sys
import time

import numpy as np

from ._base import check_X_y
from .metrics import Accuracy

__all__ = ["StreamTrainer"]


class StreamTrainer:


    def __init__(self, model, metrics=None, warm_start: bool = False):
        self.model = model
        self.metrics = metrics if metrics is not None else {"accuracy": Accuracy()}
        self.warm_start = warm_start
        self.reset()

    def reset(self) -> "StreamTrainer":
        self._cumulative = Accuracy()
        for metric in self.metrics.values():
            metric.reset()
        self._n_chunks = 0
        self.log_ = {
            "chunk_accuracy": [],
            "cumulative_accuracy": [],
            "memory_bytes": [],
            "fit_seconds": [],
        }
        for name in self.metrics:
            self.log_[name] = []
        return self

    def _classes_hint(self, y):
        return np.unique(y)

    def score_chunk(self, X, y) -> float:

        X, y = check_X_y(X, y, allow_nan=True)
        y_pred = self.model.predict(X)
        return float((y_pred == y).mean())

    def fit_chunk(self, X, y, classes=None) -> dict:

        X, y = check_X_y(X, y, allow_nan=True)
        self._n_chunks += 1
        fitted_before = self._is_fitted()

        record = {}
        if fitted_before or self.warm_start:
            y_pred = self.model.predict(X)
            chunk_acc = float((y_pred == y).mean())
            self._cumulative.update(y, y_pred)
            for metric in self.metrics.values():
                metric.update(y, y_pred)
            record["chunk_accuracy"] = chunk_acc
            record["cumulative_accuracy"] = self._cumulative.result()
            for name, metric in self.metrics.items():
                record[name] = metric.result()
        else: 
            record["chunk_accuracy"] = float("nan")
            record["cumulative_accuracy"] = float("nan")
            for name in self.metrics:
                record[name] = float("nan")

        t0 = time.perf_counter()
        self._partial_fit(X, y, classes)
        record["fit_seconds"] = time.perf_counter() - t0
        record["memory_bytes"] = self._estimate_memory()

        for key, value in record.items():
            self.log_[key].append(value)
        return record

    def run(self, chunk_iter, classes=None) -> dict:
        """Consume an iterable of ``(X_chunk, y_chunk)`` and return ``log_``."""
        for X, y in chunk_iter:
            self.fit_chunk(X, y, classes=classes)
        return self.log_


    def _partial_fit(self, X, y, classes):
        try:
            self.model.partial_fit(X, y, classes=classes)
        except TypeError:
            self.model.partial_fit(X, y)

    def _is_fitted(self) -> bool:
        return getattr(self.model, "classes_", None) is not None

    def _estimate_memory(self) -> int:

        total = sys.getsizeof(self.model)
        seen = set()

        def add(obj):
            oid = id(obj)
            if oid in seen:
                return
            seen.add(oid)
            if isinstance(obj, np.ndarray):
                nonlocal_total[0] += obj.nbytes
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    add(item)
            elif hasattr(obj, "__dict__"):
                for value in vars(obj).values():
                    add(value)

        nonlocal_total = [total]
        add(self.model)
        return int(nonlocal_total[0])

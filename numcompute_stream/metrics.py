
from __future__ import annotations

from collections import deque

import numpy as np

from ._base import BaseComponent

__all__ = [
    "Accuracy",
    "ConfusionMatrix",
    "Precision",
    "Recall",
    "F1Score",
    "ROCAUC",
    "RollingMetric",
]


def _as_pair(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true {y_true.shape} and y_pred {y_pred.shape} must have the "
            f"same shape."
        )
    return y_true, y_pred


class Accuracy(BaseComponent):


    def __init__(self):
        self.reset()

    def reset(self) -> "Accuracy":
        self._correct = 0
        self._total = 0
        return self

    def update(self, y_true, y_pred) -> "Accuracy":
        y_true, y_pred = _as_pair(y_true, y_pred)
        self._correct += int((y_true == y_pred).sum())
        self._total += y_true.size
        return self

    def result(self) -> float:
        return self._correct / self._total if self._total else 0.0


class ConfusionMatrix(BaseComponent):




    def __init__(self, labels):
        self.labels = list(labels)
        self._index = {label: i for i, label in enumerate(self.labels)}
        self.reset()

    def reset(self) -> "ConfusionMatrix":
        k = len(self.labels)
        self.matrix_ = np.zeros((k, k), dtype=np.int64)
        return self

    def update(self, y_true, y_pred) -> "ConfusionMatrix":
        y_true, y_pred = _as_pair(y_true, y_pred)
        for t, p in zip(y_true, y_pred):
            if t in self._index and p in self._index:
                self.matrix_[self._index[t], self._index[p]] += 1
        return self

    def result(self) -> np.ndarray:
        return self.matrix_.copy()


class _PerClassRate(BaseComponent):


    def __init__(self, labels, average: str = "macro"):
        if average not in {"macro", "micro", "weighted"}:
            raise ValueError(f"Unknown average {average!r}.")
        self.labels = list(labels)
        self.average = average
        self._cm = ConfusionMatrix(self.labels)
        self.reset()

    def reset(self):
        self._cm.reset()
        return self

    def update(self, y_true, y_pred):
        self._cm.update(y_true, y_pred)
        return self

    def _tp_fp_fn(self):
        m = self._cm.matrix_
        tp = np.diag(m).astype(float)
        fp = m.sum(axis=0) - tp
        fn = m.sum(axis=1) - tp
        return tp, fp, fn

    def _reduce(self, numerator_per_class, tp, fp, fn, support):
        if self.average == "micro":
            return float(numerator_per_class) 
        weights = support / support.sum() if (self.average == "weighted" and support.sum()) else None
        per_class = numerator_per_class
        if weights is not None:
            return float(np.sum(per_class * weights))
        return float(np.mean(per_class))


class Precision(_PerClassRate):


    def result(self) -> float:
        tp, fp, fn = self._tp_fp_fn()
        support = tp + fn
        if self.average == "micro":
            denom = tp.sum() + fp.sum()
            return float(tp.sum() / denom) if denom else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        return self._reduce(per_class, tp, fp, fn, support)


class Recall(_PerClassRate):


    def result(self) -> float:
        tp, fp, fn = self._tp_fp_fn()
        support = tp + fn
        if self.average == "micro":
            denom = tp.sum() + fn.sum()
            return float(tp.sum() / denom) if denom else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        return self._reduce(per_class, tp, fp, fn, support)


class F1Score(_PerClassRate):


    def result(self) -> float:
        tp, fp, fn = self._tp_fp_fn()
        support = tp + fn
        if self.average == "micro":
            p_denom = tp.sum() + fp.sum()
            r_denom = tp.sum() + fn.sum()
            p = tp.sum() / p_denom if p_denom else 0.0
            r = tp.sum() / r_denom if r_denom else 0.0
            return float(2 * p * r / (p + r)) if (p + r) else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
            rec = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
            per_class = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
        return self._reduce(per_class, tp, fp, fn, support)


class ROCAUC(BaseComponent):

    def __init__(self, capacity: int = 10000, positive_label: int = 1):
        self.capacity = capacity
        self.positive_label = positive_label
        self.reset()

    def reset(self) -> "ROCAUC":
        self._scores = np.empty(0)
        self._labels = np.empty(0, dtype=int)
        self._seen = 0         
        self._rng = np.random.default_rng(0)
        return self

    def update(self, y_true, y_score) -> "ROCAUC":
        y_true, y_score = _as_pair(y_true, y_score)
        labels = (y_true == self.positive_label).astype(int)


        if self._scores.size < self.capacity:
            free = self.capacity - self._scores.size
            take = min(free, y_score.size)
            self._scores = np.concatenate([self._scores, y_score[:take]])
            self._labels = np.concatenate([self._labels, labels[:take]])
            self._seen += take
            rest_s, rest_l = y_score[take:], labels[take:]
        else:
            rest_s, rest_l = y_score, labels

  
        for s, l in zip(rest_s, rest_l):
            self._seen += 1
            j = int(self._rng.integers(0, self._seen))
            if j < self.capacity:
                self._scores[j] = s
                self._labels[j] = l
        return self

    def result(self) -> float:
        pos = self._scores[self._labels == 1]
        neg = self._scores[self._labels == 0]
        if pos.size == 0 or neg.size == 0:
            return 0.5  
        order = np.argsort(self._scores, kind="mergesort")
        ranks = np.empty(self._scores.size)
        ranks[order] = self._rank_with_ties(self._scores[order])
        rank_pos = ranks[self._labels == 1].sum()
        auc = (rank_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
        return float(auc)

    @staticmethod
    def _rank_with_ties(sorted_scores) -> np.ndarray:

        n = sorted_scores.size
        ranks = np.arange(1, n + 1, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
                j += 1
            if j > i:
                ranks[i:j + 1] = (i + 1 + j + 1) / 2
            i = j + 1
        return ranks


class RollingMetric(BaseComponent):

    def __init__(self, metric_factory, window: int = 200):
        self.metric_factory = metric_factory
        self.window = window
        self.reset()

    def reset(self) -> "RollingMetric":
        self._true = deque(maxlen=self.window)
        self._pred = deque(maxlen=self.window)
        return self

    def update(self, y_true, y_pred) -> "RollingMetric":
        y_true, y_pred = _as_pair(y_true, y_pred)
        self._true.extend(y_true.tolist())
        self._pred.extend(y_pred.tolist())
        return self

    def result(self):
        metric = self.metric_factory()
        if self._true:
            metric.update(np.array(self._true), np.array(self._pred))
        return metric.result()

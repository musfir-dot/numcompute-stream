
from __future__ import annotations

import matplotlib

if matplotlib.get_backend().lower() not in {"agg", "module://matplotlib_inline.backend_inline"}:
    try:  
        matplotlib.use("Agg")
    except Exception:
        pass

import matplotlib.pyplot as plt  
import numpy as np  

__all__ = [
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
    "plot_confusion_matrix",
]


def _finish(fig, ax, save_path, show):
    """Shared save/show/return handling for the plotting helpers."""
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def plot_metric_over_time(metric_values, *, title: str = "Metric over time",
                          ylabel: str = "value", xlabel: str = "chunk",
                          ax=None, save_path: str | None = None, show: bool = False):

    values = np.asarray(metric_values, dtype=float)
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    fig = ax.figure
    x = np.arange(1, values.size + 1)
    ax.plot(x, values, marker="o", markersize=3, linewidth=1.6, color="#2563eb")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return _finish(fig, ax, save_path, show)


def compare_models(metric1, metric2, *, labels=("model A", "model B"),
                   title: str = "Model comparison", ylabel: str = "value",
                   xlabel: str = "chunk", ax=None, save_path: str | None = None,
                   show: bool = False):

    m1 = np.asarray(metric1, dtype=float)
    m2 = np.asarray(metric2, dtype=float)
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    fig = ax.figure
    ax.plot(np.arange(1, m1.size + 1), m1, marker="o", markersize=3,
            label=labels[0], color="#2563eb", linewidth=1.6)
    ax.plot(np.arange(1, m2.size + 1), m2, marker="s", markersize=3,
            label=labels[1], color="#dc2626", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finish(fig, ax, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, *, title: str = "Predictions vs ground truth",
                                     ax=None, save_path: str | None = None, show: bool = False):

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    fig = ax.figure
    idx = np.arange(y_true.size)
    ax.scatter(idx, y_true, marker="o", s=28, label="true", color="#2563eb", alpha=0.7)
    ax.scatter(idx, y_pred, marker="x", s=36, label="pred", color="#dc2626")
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    ax.set_title(f"{title}  (acc={acc:.3f})")
    ax.set_xlabel("sample")
    ax.set_ylabel("class label")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finish(fig, ax, save_path, show)


def plot_confusion_matrix(matrix, *, labels=None, title: str = "Confusion matrix",
                          ax=None, save_path: str | None = None, show: bool = False):
    matrix = np.asarray(matrix)
    ax = ax or plt.subplots(figsize=(5, 4.5))[1]
    fig = ax.figure
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    k = matrix.shape[0]
    ticks = labels if labels is not None else np.arange(k)
    ax.set_xticks(range(k), ticks)
    ax.set_yticks(range(k), ticks)
    thresh = matrix.max() / 2 if matrix.max() else 0
    for i in range(k):
        for j in range(k):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black")
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    return _finish(fig, ax, save_path, show)

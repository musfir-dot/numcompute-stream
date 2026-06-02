import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from numcompute_stream.io import load_csv, iter_chunks
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.metrics import F1Score, ConfusionMatrix

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

NAVY = "#1a2b4a"
ORANGE = "#d9772b"
GREY = "#888888"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cccccc",
                     "axes.grid": True, "grid.color": "#e8e8e8",
                     "figure.dpi": 130})



data = load_csv(os.path.join(HERE, "..", "demo", "stream_data.csv"), target="label")
X, y = data["X"], data["y"].astype(int)
classes = np.unique(y)


def make_pipe(model):
    return Pipeline([("impute", Imputer(strategy="mean")),
                     ("scale", StandardScaler()), ("model", model)])


rf_tr = StreamTrainer(make_pipe(RandomForestClassifier(
    n_estimators=20, max_depth=10, random_state=0, max_buffer=3000)),
    metrics={"f1": F1Score(labels=classes.tolist(), average="macro")})
tree_tr = StreamTrainer(make_pipe(DecisionTreeClassifier(max_depth=10, max_buffer=3000)),
    metrics={"f1": F1Score(labels=classes.tolist(), average="macro")})

N = 20
for Xc, yc in iter_chunks(X, y, n_chunks=N):
    rf_tr.fit_chunk(Xc, yc, classes=classes)
    tree_tr.fit_chunk(Xc, yc, classes=classes)
rf, tr = rf_tr.log_, tree_tr.log_
chunks_x = np.arange(1, len(rf["cumulative_accuracy"]) + 1)


fig, ax = plt.subplots(figsize=(4.6, 2.7))
ax.plot(chunks_x, rf["cumulative_accuracy"], "-o", color=NAVY, ms=3.5,
        lw=1.8, label="random forest")
ax.plot(chunks_x, tr["cumulative_accuracy"], "-s", color=ORANGE, ms=3.5,
        lw=1.8, label="single tree")
ax.set_xlabel("chunk"); ax.set_ylabel("cumulative accuracy")
ax.set_title("Prequential cumulative accuracy", fontsize=10.5, color=NAVY)
ax.legend(frameon=False, fontsize=8.5); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_cumacc.png")); plt.close(fig)


fig, ax = plt.subplots(figsize=(4.6, 2.7))
ax.plot(chunks_x, rf["chunk_accuracy"], "-o", color=NAVY, ms=3.5, lw=1.6,
        label="random forest")
ax.plot(chunks_x, tr["chunk_accuracy"], "-s", color=ORANGE, ms=3.5, lw=1.6,
        label="single tree")
ax.set_xlabel("chunk"); ax.set_ylabel("per-chunk accuracy")
ax.set_title("Per-chunk accuracy (lower variance = ensemble)", fontsize=10.5,
             color=NAVY)
ax.legend(frameon=False, fontsize=8.5); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_chunkacc.png")); plt.close(fig)


fig, ax = plt.subplots(figsize=(4.6, 2.7))
ax.plot(chunks_x, [m / 1024 for m in rf["memory_bytes"]], "-o", color=NAVY,
        ms=3.5, lw=1.8)
ax.set_xlabel("chunk"); ax.set_ylabel("training buffer (KB)")
ax.set_ylim(0, max([m / 1024 for m in rf["memory_bytes"]]) * 1.5)
ax.set_title("Memory stays bounded on an unbounded stream", fontsize=10.5,
             color=NAVY); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_memory.png")); plt.close(fig)


Xc, yc = list(iter_chunks(X, y, n_chunks=N))[-1]
y_pred = rf_tr.model.predict(Xc)
cm = ConfusionMatrix(labels=classes.tolist()); cm.update(yc, y_pred)
M = cm.result()
fig, ax = plt.subplots(figsize=(3.0, 2.7))
im = ax.imshow(M, cmap="Blues")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, int(M[i, j]), ha="center", va="center",
                color="white" if M[i, j] > M.max() / 2 else NAVY, fontsize=11)
ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes); ax.set_yticklabels(classes)
ax.set_xlabel("predicted"); ax.set_ylabel("actual")
ax.set_title("Confusion matrix (last chunk)", fontsize=10.5, color=NAVY)
ax.grid(False); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_confusion.png")); plt.close(fig)



def _gini(counts):
    t = counts.sum()
    if t == 0:
        return 0.0
    p = counts / t
    return 1.0 - np.sum(p * p)


def naive_split(X, y):
    n, nf = X.shape
    cls = np.unique(y)
    pc = np.array([(y == c).sum() for c in cls], float)
    pim = _gini(pc); tot = float(n); bg = 0.0
    for f in range(nf):
        col = X[:, f]; order = np.argsort(col); xs, ys = col[order], y[order]
        for i in range(n - 1):
            if xs[i] == xs[i + 1]:
                continue
            lc = np.array([(ys[:i + 1] == c).sum() for c in cls], float)
            rc = np.array([(ys[i + 1:] == c).sum() for c in cls], float)
            w = (lc.sum() * _gini(lc) + rc.sum() * _gini(rc)) / tot
            bg = max(bg, pim - w)
    return bg


def vec_split(X, y):
    t = DecisionTreeClassifier(criterion="gini"); t.classes_ = np.unique(y)
    return t._best_split(X, y, np.ones(len(y)))


rng = np.random.default_rng(0)
sizes = [500, 1000, 2000, 4000]
naive_t, vec_t = [], []
for n in sizes:
    Xs = rng.normal(size=(n, 8)); ys = (Xs[:, 0] + Xs[:, 1] > 0).astype(int)
    t0 = time.perf_counter(); naive_split(Xs, ys); naive_t.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); vec_split(Xs, ys); vec_t.append(time.perf_counter() - t0)

fig, ax = plt.subplots(figsize=(4.6, 2.8))
xb = np.arange(len(sizes)); w = 0.38
ax.bar(xb - w / 2, naive_t, w, color=ORANGE, label="naive Python loop")
ax.bar(xb + w / 2, vec_t, w, color=NAVY, label="vectorised (ours)")
ax.set_yscale("log"); ax.set_xticks(xb); ax.set_xticklabels(sizes)
ax.set_xlabel("samples at node"); ax.set_ylabel("time (s, log scale)")
ax.set_title("Split-finding: vectorised vs. loop", fontsize=10.5, color=NAVY)
for i, (nt, vt) in enumerate(zip(naive_t, vec_t)):
    ax.text(i, max(naive_t) * 1.3, f"{nt / vt:.0f}x", ha="center",
            fontsize=8.5, color=NAVY, fontweight="bold")
ax.legend(frameon=False, fontsize=8.5); ax.grid(axis="x"); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_speedup.png")); plt.close(fig)



def make_stream(seed=0, n=12000, d=10):
    r = np.random.default_rng(seed); Xx = r.normal(size=(n, d))
    yy = ((Xx[:, 0] * Xx[:, 1] + np.sin(2 * Xx[:, 2]) - Xx[:, 3] ** 2) > 0).astype(int)
    return Xx, yy


Xs, ys = make_stream()
labels_m = ["single\ntree", "online\nbagging", "random\nforest"]
models_m = [DecisionTreeClassifier(max_depth=10, max_buffer=4000),
            RandomForestClassifier(n_estimators=20, max_depth=10, random_state=0,
                                   streaming_mode="online_bagging", max_buffer=4000),
            RandomForestClassifier(n_estimators=20, max_depth=10, random_state=0,
                                   max_buffer=4000)]
accs, times = [], []
for m in models_m:
    tr2 = StreamTrainer(m)
    t0 = time.perf_counter()
    for idx in np.array_split(np.arange(len(ys)), 30):
        tr2.fit_chunk(Xs[idx], ys[idx], classes=[0, 1])
    times.append(time.perf_counter() - t0)
    accs.append(tr2.log_["cumulative_accuracy"][-1])

fig, ax1 = plt.subplots(figsize=(4.6, 2.8))
xb = np.arange(3)
b1 = ax1.bar(xb - 0.2, accs, 0.4, color=NAVY, label="cum. accuracy")
ax1.set_ylim(0.8, 0.9); ax1.set_ylabel("cumulative accuracy", color=NAVY)
ax1.set_xticks(xb); ax1.set_xticklabels(labels_m, fontsize=8.5)
ax2 = ax1.twinx(); ax2.grid(False)
b2 = ax2.bar(xb + 0.2, times, 0.4, color=ORANGE, label="fit time")
ax2.set_ylabel("total fit time (s)", color=ORANGE)
ax1.set_title("Base vs. ensemble: accuracy vs. cost", fontsize=10.5, color=NAVY)
for i, a in enumerate(accs):
    ax1.text(i - 0.2, a + 0.002, f"{a:.3f}", ha="center", fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_tradeoff.png")); plt.close(fig)

print("RF cum.acc", round(rf["cumulative_accuracy"][-1], 4),
      "| tree", round(tr["cumulative_accuracy"][-1], 4))
print("speedups", [round(n / v) for n, v in zip(naive_t, vec_t)])
print("ensemble accs", [round(a, 4) for a in accs], "times", [round(t, 1) for t in times])
print("figures:", sorted(os.listdir(FIG)))

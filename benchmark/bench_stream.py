
from __future__ import annotations

import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier, RandomForestClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.metrics import F1Score


def make_stream(seed=0, n=12000, d=10):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = ((X[:, 0] * X[:, 1] + np.sin(2 * X[:, 2]) - X[:, 3] ** 2) > 0).astype(int)
    return X, y


def run_model(name, model, X, y, n_chunks=30):
    trainer = StreamTrainer(model, metrics={"f1": F1Score(labels=[0, 1], average="macro")})
    chunks = np.array_split(np.arange(len(y)), n_chunks)
    for idx in chunks:
        trainer.fit_chunk(X[idx], y[idx], classes=[0, 1])
    log = trainer.log_
    return {
        "name": name,
        "cum_acc": log["cumulative_accuracy"][-1],
        "f1": log["f1"][-1],
        "fit_s": float(np.sum(log["fit_seconds"])),
        "mem_kb": log["memory_bytes"][-1] / 1024.0,
    }


def main() -> None:
    X, y = make_stream()
    models = [
        ("Single tree", DecisionTreeClassifier(max_depth=10, max_buffer=4000)),
        ("Ensemble (online bagging)",
         EnsembleClassifier(n_estimators=20, max_depth=10, random_state=0,
                            streaming_mode="online_bagging", max_buffer=4000)),
        ("Random forest",
         RandomForestClassifier(n_estimators=20, max_depth=10, random_state=0,
                                max_buffer=4000)),
    ]

    print(f"{'model':>26} | {'cum.acc':>7} | {'macroF1':>7} | "
          f"{'fit(s)':>7} | {'mem(KB)':>8}")
    print("-" * 70)
    for name, model in models:
        r = run_model(name, model, X, y)
        print(f"{r['name']:>26} | {r['cum_acc']:>7.4f} | {r['f1']:>7.4f} | "
              f"{r['fit_s']:>7.2f} | {r['mem_kb']:>8.1f}")


if __name__ == "__main__":
    main()

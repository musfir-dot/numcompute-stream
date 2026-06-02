# NumCompute-Stream

A modularised, ensemble **tree-based streaming** machine-learning framework,
built **from scratch using only NumPy and matplotlib** (no scikit-learn, pandas
or PyTorch). Every component (preprocessing, statistics, metrics, models and
the pipeline) supports incremental, chunk-wise updates through a small shared
interface, so they compose cleanly and can be driven by a single streaming
trainer.

---

## Highlights

- **Streaming-first**: every component exposes `partial_fit` / `update` and
  produces results that are invariant to how the data is chunked.
- **Vectorised core**: split finding, streaming moments and metrics are
  implemented with NumPy array operations rather than Python loops
  (50 to 125x faster split search, see `benchmark/`).
- **Numerically stable**: parallel Welford for mean/variance, the P² algorithm
  for O(1)-memory quantiles, tie-averaged ranks for AUC, eps-floored scaling for
  zero-variance features, and per-cell `NaN` handling throughout.
- **Ensembles**: bagging, Oza online bagging and random forests built from the
  same `DecisionTreeClassifier`, switchable behind a shared interface.
- **Bounded memory**: trees train from a sliding buffer, so memory stays
  constant on an unbounded stream while still adapting to concept drift.

## Installation

No build step is required; the package is pure Python + NumPy + matplotlib.

```bash
pip install numpy matplotlib
cd numcompute_project
python -m unittest discover -s tests
```

Add the project root to `PYTHONPATH` (or work from it) so that
`import numcompute_stream` resolves.

## Package layout

```
numcompute_stream/
  _base.py         shared interfaces (StreamEstimator/Transformer) + validators
  stats.py         RunningMoments, StreamingQuantile, StreamingHistogram, update_stats
  preprocessing.py StandardScaler, Imputer, OneHotEncoder
  metrics.py       Accuracy, Precision, Recall, F1Score, ConfusionMatrix, ROCAUC, RollingMetric
  tree.py          DecisionTreeClassifier (vectorised CART, streaming buffer)
  ensemble.py      EnsembleClassifier, RandomForestClassifier
  pipeline.py      Pipeline (chains transformers + estimator under partial_fit)
  stream.py        StreamTrainer (prequential test-then-train orchestration)
  io.py            load_csv, iter_chunks, train_test_split
  visualise.py     plot_metric_over_time, compare_models, plot_predictions_vs_ground_truth, plot_confusion_matrix
tests/             84 unit tests (standard + edge cases under streaming)
benchmark/         loop-vs-vectorised and base-vs-ensemble benchmarks
demo/              stream_demo.ipynb + dataset generator
report/            technical report (PDF)
```

## Quick start

```python
import numpy as np
from numcompute_stream import (
    Pipeline, Imputer, StandardScaler, RandomForestClassifier,
    StreamTrainer, F1Score, iter_chunks,
)

pipe = Pipeline([
    ("impute", Imputer(strategy="mean")),
    ("scale",  StandardScaler()),
    ("model",  RandomForestClassifier(n_estimators=20, max_depth=10)),
])

trainer = StreamTrainer(pipe, metrics={"f1": F1Score(labels=[0, 1])})

for X_chunk, y_chunk in iter_chunks(X, y, n_chunks=20):
    trainer.fit_chunk(X_chunk, y_chunk, classes=[0, 1])

log = trainer.log_
print(log["cumulative_accuracy"][-1], log["f1"][-1])
```

### Streaming statistics

```python
from numcompute_stream import RunningMoments, StreamingQuantile

rm = RunningMoments()
for chunk in chunks:
    rm.update(chunk)
print(rm.mean_, rm.variance_)

q = StreamingQuantile(0.9)
for chunk in chunks:
    q.update(chunk.ravel())
print(q.result())
```

### Metrics with rolling windows

```python
from numcompute_stream import Accuracy, RollingMetric

acc = Accuracy()
roll = RollingMetric(lambda: Accuracy(), window=200)
for yt, yp in stream_of_predictions:
    acc.update(yt, yp); roll.update(yt, yp)
print(acc.result(), roll.result())
```

## Running the demo

```bash
cd demo
python make_dataset.py
jupyter notebook stream_demo.ipynb
```

The notebook loads the CSV via `io.py`, splits it into chunks, trains the
pipeline incrementally, and visualises accuracy, memory and predictions over
time, comparing the ensemble against a single tree.

## Running the benchmarks

```bash
python -m benchmark.bench_split
python -m benchmark.bench_stats
python -m benchmark.bench_stream
```

Captured results are in `benchmark/results.txt`.

## Design notes

- **Shared interface (`_base.py`)**: `StreamTransformer` (`partial_fit` +
  `transform`) and `StreamEstimator` (`partial_fit` + `predict`) let `Pipeline`
  and `StreamTrainer` treat every component uniformly, which is what makes the
  framework modular and reusable.
- **Trees as batch-incremental learners**: `partial_fit` appends the chunk to a
  bounded buffer, evicts the oldest rows beyond `max_buffer`, and re-induces.
  This keeps memory bounded, adapts to drift, and yields a tree identical to the
  batch tree over the retained window.
- **Genuine streaming ensemble**: `EnsembleClassifier(streaming_mode="online_bagging")`
  implements Oza online bagging, weighting each sample by a `Poisson(1)` draw per
  tree to emulate bootstrap resampling without storing the full dataset.

See `report/` for the full technical write-up.

## Testing

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

84 tests cover standard behaviour and edge cases: `NaN` inputs, ties,
zero-variance chunks, streaming-vs-batch equivalence, shape/type validation, and
ensemble-beats-tree generalisation.
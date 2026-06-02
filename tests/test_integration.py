import os
import tempfile
import unittest

import numpy as np

from numcompute_stream.ensemble import EnsembleClassifier, RandomForestClassifier
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler, Imputer
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.metrics import Accuracy, F1Score
from numcompute_stream.io import load_csv, iter_chunks, train_test_split


def nonlinear(seed=0, n=3000, d=8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = ((X[:, 0] * X[:, 1] + np.sin(X[:, 2]) - X[:, 3] ** 2) > 0).astype(int)
    return X, y


class TestEnsemble(unittest.TestCase):
    def test_forest_fits(self):
        X, y = nonlinear()
        rf = RandomForestClassifier(n_estimators=15, max_depth=8, random_state=0).fit(X, y)
        self.assertGreater((rf.predict(X) == y).mean(), 0.85)

    def test_forest_beats_single_tree(self):

        tree_accs, forest_accs = [], []
        for seed in range(5):
            X, y = nonlinear(seed=seed, n=4000, d=10)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=seed)
            tree = DecisionTreeClassifier(max_depth=14).fit(Xtr, ytr)
            rf = RandomForestClassifier(
                n_estimators=40, max_depth=14, random_state=seed).fit(Xtr, ytr)
            tree_accs.append((tree.predict(Xte) == yte).mean())
            forest_accs.append((rf.predict(Xte) == yte).mean())
        self.assertGreater(np.mean(forest_accs), np.mean(tree_accs))

    def test_online_bagging_streaming(self):
        X, y = nonlinear()
        rf = RandomForestClassifier(n_estimators=15, max_depth=8, random_state=0, max_buffer=2000)
        for c in np.array_split(np.arange(len(y)), 20):
            rf.partial_fit(X[c], y[c], classes=[0, 1])
        self.assertGreater((rf.predict(X) == y).mean(), 0.8)

    def test_rebuild_mode_streaming(self):
        X, y = nonlinear(n=2000)
        ens = EnsembleClassifier(n_estimators=10, max_depth=6,
                                 streaming_mode="rebuild", random_state=1, max_buffer=2000)
        for c in np.array_split(np.arange(len(y)), 10):
            ens.partial_fit(X[c], y[c])
        self.assertGreater((ens.predict(X) == y).mean(), 0.75)

    def test_proba_sums_to_one(self):
        X, y = nonlinear(n=500)
        rf = RandomForestClassifier(n_estimators=8, random_state=0).fit(X, y)
        np.testing.assert_allclose(rf.predict_proba(X).sum(1), 1.0, atol=1e-9)

    def test_invalid_streaming_mode_raises(self):
        with self.assertRaises(ValueError):
            EnsembleClassifier(streaming_mode="magic")


class TestPipeline(unittest.TestCase):
    def test_chained_partial_fit_predict(self):
        X, y = nonlinear(n=2000)
        pipe = Pipeline([("scale", StandardScaler()),
                         ("model", DecisionTreeClassifier(max_depth=6, max_buffer=2000))])
        for c in np.array_split(np.arange(len(y)), 10):
            pipe.partial_fit(X[c], y[c])
        self.assertGreater((pipe.predict(X) == y).mean(), 0.8)

    def test_imputer_in_pipeline(self):
        X, y = nonlinear(n=1500)
        Xm = X.copy()
        Xm[np.random.default_rng(0).random(X.shape) < 0.05] = np.nan
        pipe = Pipeline([("impute", Imputer("mean")),
                         ("scale", StandardScaler()),
                         ("model", RandomForestClassifier(n_estimators=10, random_state=0, max_buffer=1500))])
        for c in np.array_split(np.arange(len(y)), 10):
            pipe.partial_fit(Xm[c], y[c], classes=[0, 1])
        self.assertFalse(np.isnan(pipe.predict(Xm)).any())

    def test_duplicate_step_names_raise(self):
        with self.assertRaises(ValueError):
            Pipeline([("a", StandardScaler()), ("a", StandardScaler())])

    def test_named_steps_access(self):
        pipe = Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier())])
        self.assertIn("scale", pipe.named_steps)

    def test_transform_only_pipeline(self):
        X, _ = nonlinear(n=300)
        pipe = Pipeline([("scale", StandardScaler())])
        pipe.partial_fit(X)
        Z = pipe.transform(X)
        np.testing.assert_allclose(Z.mean(0), 0, atol=1e-6)


class TestStreamTrainer(unittest.TestCase):
    def test_prequential_run(self):
        X, y = nonlinear(n=3000)
        model = RandomForestClassifier(n_estimators=12, random_state=0, max_buffer=2000)
        trainer = StreamTrainer(model, metrics={"acc": Accuracy(), "f1": F1Score([0, 1])})
        log = trainer.run(iter_chunks(X, y, n_chunks=15), classes=[0, 1])
        self.assertEqual(len(log["chunk_accuracy"]), 15)
        self.assertGreater(log["cumulative_accuracy"][-1], 0.7)

    def test_first_chunk_is_train_only(self):
        X, y = nonlinear(n=600)
        trainer = StreamTrainer(DecisionTreeClassifier(max_buffer=600))
        log = trainer.run(iter_chunks(X, y, n_chunks=6))
        self.assertTrue(np.isnan(log["chunk_accuracy"][0]))
        self.assertFalse(np.isnan(log["chunk_accuracy"][-1]))

    def test_memory_logged_and_bounded(self):
        X, y = nonlinear(n=3000)
        trainer = StreamTrainer(DecisionTreeClassifier(max_depth=5, max_buffer=500))
        log = trainer.run(iter_chunks(X, y, n_chunks=20))
        self.assertEqual(len(log["memory_bytes"]), 20)
        # later memory should not blow up versus mid-stream (bounded buffer)
        self.assertLessEqual(log["memory_bytes"][-1], log["memory_bytes"][10] * 2)

    def test_score_chunk(self):
        X, y = nonlinear(n=500)
        model = DecisionTreeClassifier(max_buffer=500).fit(X, y)
        trainer = StreamTrainer(model)
        self.assertGreaterEqual(trainer.score_chunk(X, y), 0.8)


class TestIO(unittest.TestCase):
    def _write(self, text):
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        return path

    def test_numeric_csv(self):
        path = self._write("a,b,label\n1,2,0\n3,4,1\n5,6,0\n")
        data = load_csv(path, target="label")
        self.assertEqual(data["X"].shape, (3, 2))
        np.testing.assert_array_equal(data["y"], [0, 1, 0])
        os.remove(path)

    def test_categorical_encoded(self):
        path = self._write("color,label\nred,0\nblue,1\nred,0\n")
        data = load_csv(path, target="label")
        self.assertEqual(data["X"].shape, (3, 1))
        self.assertIn(0, data["encoders"])  # 'color' column encoded
        os.remove(path)

    def test_missing_values_become_nan(self):
        path = self._write("a,label\n1,0\n,1\n3,0\n")
        data = load_csv(path, target="label")
        self.assertTrue(np.isnan(data["X"]).any())
        os.remove(path)

    def test_iter_chunks_covers_all(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        total = sum(len(yc) for _, yc in iter_chunks(X, y, n_chunks=7))
        self.assertEqual(total, 50)

    def test_train_test_split_sizes(self):
        X = np.arange(200).reshape(100, 2)
        y = np.arange(100)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        self.assertEqual(len(Xte), 30)
        self.assertEqual(len(Xtr), 70)


if __name__ == "__main__":
    unittest.main()

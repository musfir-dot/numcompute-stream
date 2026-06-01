import unittest

import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier


def make_data(seed=0, n=2000, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.6 * X[:, 1] - 0.4 * X[:, 2] > 0).astype(int)
    return X, y


class TestDecisionTree(unittest.TestCase):
    def test_fits_separable_data(self):
        X, y = make_data()
        t = DecisionTreeClassifier(max_depth=8).fit(X, y)
        self.assertGreater((t.predict(X) == y).mean(), 0.9)

    def test_gini_and_entropy_both_work(self):
        X, y = make_data()
        for crit in ("gini", "entropy"):
            t = DecisionTreeClassifier(max_depth=6, criterion=crit).fit(X, y)
            self.assertGreater((t.predict(X) == y).mean(), 0.85)

    def test_depth_respected(self):
        X, y = make_data()
        t = DecisionTreeClassifier(max_depth=3).fit(X, y)
        self.assertLessEqual(t.depth(), 3)

    def test_proba_sums_to_one(self):
        X, y = make_data(n=500)
        t = DecisionTreeClassifier().fit(X, y)
        np.testing.assert_allclose(t.predict_proba(X).sum(1), 1.0, atol=1e-9)

    def test_pure_node_single_class(self):
        X = np.random.default_rng(0).normal(size=(50, 3))
        y = np.zeros(50, dtype=int)
        t = DecisionTreeClassifier().fit(X, y)
        self.assertTrue((t.predict(X) == 0).all())

    def test_streaming_matches_full_buffer(self):
        X, y = make_data(n=1500)
        stream = DecisionTreeClassifier(max_depth=6, max_buffer=10000)
        for c in np.array_split(np.arange(1500), 15):
            stream.partial_fit(X[c], y[c])
        batch = DecisionTreeClassifier(max_depth=6).fit(X, y)
        # Same data in buffer -> identical predictions.
        np.testing.assert_array_equal(stream.predict(X), batch.predict(X))

    def test_sliding_buffer_bounds_memory(self):
        X, y = make_data(n=3000)
        t = DecisionTreeClassifier(max_depth=5, max_buffer=500)
        for c in np.array_split(np.arange(3000), 30):
            t.partial_fit(X[c], y[c])
        self.assertEqual(t._Xbuf.shape[0], 500)

    def test_max_features_sqrt(self):
        X, y = make_data(d=16)
        t = DecisionTreeClassifier(max_depth=5, max_features="sqrt", random_state=0).fit(X, y)
        self.assertGreater((t.predict(X) == y).mean(), 0.7)

    def test_sample_weight_zero_excludes_class(self):
        X, y = make_data()
        w = np.ones(len(y))
        w[y == 1] = 0.0
        t = DecisionTreeClassifier(max_depth=4).fit(X, y, sample_weight=w)
        self.assertTrue((t.predict(X) == 0).all())

    def test_nan_input_raises(self):
        X, y = make_data(n=20)
        X[0, 0] = np.nan
        with self.assertRaises(ValueError):
            DecisionTreeClassifier().fit(X, y)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            DecisionTreeClassifier().predict(np.zeros((2, 3)))

    def test_constant_feature_no_crash(self):
        X = np.ones((100, 3))
        X[:, 1] = np.random.default_rng(0).normal(size=100)
        y = (X[:, 1] > 0).astype(int)
        t = DecisionTreeClassifier(max_depth=4).fit(X, y)
        self.assertGreater((t.predict(X) == y).mean(), 0.9)

    def test_tied_feature_values(self):
        # Many identical rows -> no valid split should not crash.
        X = np.zeros((40, 2))
        y = np.array([0, 1] * 20)
        t = DecisionTreeClassifier(max_depth=5).fit(X, y)
        self.assertEqual(t.predict(X).shape, (40,))

    def test_invalid_criterion_raises(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(criterion="huber")


if __name__ == "__main__":
    unittest.main()

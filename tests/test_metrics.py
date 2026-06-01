import unittest

import numpy as np

from numcompute_stream.metrics import (
    Accuracy, ConfusionMatrix, Precision, Recall, F1Score, ROCAUC, RollingMetric,
)


class TestAccuracy(unittest.TestCase):
    def test_perfect(self):
        acc = Accuracy().update([0, 1, 2], [0, 1, 2])
        self.assertEqual(acc.result(), 1.0)

    def test_streaming_equals_batch(self):
        rng = np.random.default_rng(0)
        yt = rng.integers(0, 4, 1000)
        yp = rng.integers(0, 4, 1000)
        acc = Accuracy()
        for c in np.array_split(np.arange(1000), 20):
            acc.update(yt[c], yp[c])
        self.assertAlmostEqual(acc.result(), (yt == yp).mean())

    def test_empty_result_zero(self):
        self.assertEqual(Accuracy().result(), 0.0)

    def test_reset(self):
        acc = Accuracy().update([0], [0])
        acc.reset()
        self.assertEqual(acc.result(), 0.0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            Accuracy().update([0, 1], [0])


class TestConfusionMatrix(unittest.TestCase):
    def test_counts(self):
        cm = ConfusionMatrix([0, 1]).update([0, 0, 1, 1], [0, 1, 1, 1])
        np.testing.assert_array_equal(cm.matrix_, [[1, 1], [0, 2]])

    def test_accumulates(self):
        cm = ConfusionMatrix([0, 1])
        cm.update([0], [0])
        cm.update([1], [1])
        self.assertEqual(cm.matrix_.sum(), 2)

    def test_unknown_labels_skipped(self):
        cm = ConfusionMatrix([0, 1]).update([0, 9], [0, 9])
        self.assertEqual(cm.matrix_.sum(), 1)


class TestPRF(unittest.TestCase):
    def setUp(self):
        self.yt = np.array([0, 0, 1, 1, 2, 2])
        self.yp = np.array([0, 1, 1, 1, 2, 0])

    def test_macro_recall(self):
        r = Recall([0, 1, 2], "macro").update(self.yt, self.yp)
        # recalls: class0 1/2, class1 2/2, class2 1/2 -> mean = 2/3
        self.assertAlmostEqual(r.result(), (0.5 + 1.0 + 0.5) / 3)

    def test_micro_equals_accuracy(self):
        p = Precision([0, 1, 2], "micro").update(self.yt, self.yp)
        self.assertAlmostEqual(p.result(), (self.yt == self.yp).mean())

    def test_f1_between_p_and_r(self):
        for avg in ("macro", "micro", "weighted"):
            f = F1Score([0, 1, 2], avg).update(self.yt, self.yp)
            self.assertGreaterEqual(f.result(), 0.0)
            self.assertLessEqual(f.result(), 1.0)

    def test_invalid_average_raises(self):
        with self.assertRaises(ValueError):
            Precision([0, 1], "nonsense")


class TestROCAUC(unittest.TestCase):
    def _ref(self, label, score):
        order = np.argsort(score)
        ranks = np.empty(len(score), dtype=float)
        ranks[order] = np.arange(1, len(score) + 1)
        pos = label == 1
        npos, nneg = pos.sum(), (~pos).sum()
        return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)

    def test_matches_reference(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 4000)
        s = y * 0.6 + rng.random(4000)
        auc = ROCAUC(capacity=100000)
        for c in np.array_split(np.arange(4000), 20):
            auc.update(y[c], s[c])
        self.assertAlmostEqual(auc.result(), self._ref(y, s), places=4)

    def test_perfect_separation(self):
        auc = ROCAUC(capacity=1000).update([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(auc.result(), 1.0)

    def test_single_class_returns_half(self):
        auc = ROCAUC().update([0, 0, 0], [0.1, 0.2, 0.3])
        self.assertEqual(auc.result(), 0.5)

    def test_ties_handled(self):
        auc = ROCAUC(capacity=100).update([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(auc.result(), 0.5)


class TestRollingMetric(unittest.TestCase):
    def test_window_drops_old(self):
        roll = RollingMetric(Accuracy, window=50)
        yt = np.r_[np.zeros(100), np.zeros(100)]
        yp = np.r_[np.zeros(100), np.ones(100)]  # last 100 all wrong
        for c in np.array_split(np.arange(200), 20):
            roll.update(yt[c], yp[c])
        self.assertLess(roll.result(), 0.1)

    def test_reset(self):
        roll = RollingMetric(Accuracy, window=10).update([0], [0])
        roll.reset()
        self.assertEqual(roll.result(), 0.0)


if __name__ == "__main__":
    unittest.main()

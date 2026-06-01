import unittest

import numpy as np

from numcompute_stream.stats import (
    RunningMoments, StreamingQuantile, StreamingHistogram, update_stats,
)


class TestRunningMoments(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.data = self.rng.normal(3, 5, size=(2000, 4))

    def test_mean_matches_numpy(self):
        rm = RunningMoments()
        for chunk in np.array_split(self.data, 17):
            rm.update(chunk)
        np.testing.assert_allclose(rm.mean_, self.data.mean(0), atol=1e-10)

    def test_variance_matches_numpy(self):
        rm = RunningMoments()
        for chunk in np.array_split(self.data, 17):
            rm.update(chunk)
        np.testing.assert_allclose(rm.variance_, self.data.var(0), atol=1e-9)

    def test_chunking_invariance(self):
        a = RunningMoments()
        for c in np.array_split(self.data, 3):
            a.update(c)
        b = RunningMoments()
        for c in np.array_split(self.data, 50):
            b.update(c)
        np.testing.assert_allclose(a.mean_, b.mean_, atol=1e-10)
        np.testing.assert_allclose(a.variance_, b.variance_, atol=1e-9)

    def test_nan_ignored_per_cell(self):
        d = self.data.copy()
        d[::4, 0] = np.nan
        rm = RunningMoments()
        for c in np.array_split(d, 11):
            rm.update(c)
        self.assertAlmostEqual(rm.mean_[0], np.nanmean(d[:, 0]), places=9)

    def test_zero_variance_feature(self):
        const = np.full((100, 2), 7.0)
        rm = RunningMoments().update(const)
        np.testing.assert_allclose(rm.variance_, 0.0)

    def test_feature_count_mismatch_raises(self):
        rm = RunningMoments().update(np.ones((5, 3)))
        with self.assertRaises(ValueError):
            rm.update(np.ones((5, 4)))

    def test_sample_variance_unbiased(self):
        rm = RunningMoments().update(self.data)
        np.testing.assert_allclose(rm.sample_variance_, self.data.var(0, ddof=1), atol=1e-8)


class TestStreamingQuantile(unittest.TestCase):
    def test_median_converges(self):
        rng = np.random.default_rng(1)
        data = rng.normal(0, 1, 20000)
        q = StreamingQuantile(0.5)
        for c in np.array_split(data, 100):
            q.update(c)
        self.assertAlmostEqual(q.value_, np.median(data), delta=0.05)

    def test_high_quantile(self):
        rng = np.random.default_rng(2)
        data = rng.uniform(0, 1, 20000)
        q = StreamingQuantile(0.9).update(data)
        self.assertAlmostEqual(q.value_, np.quantile(data, 0.9), delta=0.03)

    def test_invalid_q_raises(self):
        with self.assertRaises(ValueError):
            StreamingQuantile(0.0)
        with self.assertRaises(ValueError):
            StreamingQuantile(1.0)

    def test_fewer_than_five_points(self):
        q = StreamingQuantile(0.5).update([1.0, 3.0, 2.0])
        self.assertAlmostEqual(q.value_, 2.0)

    def test_nan_filtered(self):
        q = StreamingQuantile(0.5).update([1, 2, np.nan, 3, 4, 5])
        self.assertFalse(np.isnan(q.value_))


class TestStreamingHistogram(unittest.TestCase):
    def test_counts_total(self):
        h = StreamingHistogram(bins=5, value_range=(0, 1))
        h.update(np.random.default_rng(0).uniform(0, 1, 500))
        self.assertEqual(h.counts_.sum(), 500)

    def test_sliding_window_caps(self):
        h = StreamingHistogram(bins=4, value_range=(0, 1), window=100)
        h.update(np.random.default_rng(0).uniform(0, 1, 400))
        self.assertLessEqual(h.counts_.sum(), 100)

    def test_density_integrates_to_one(self):
        h = StreamingHistogram(bins=10, value_range=(0, 1))
        h.update(np.random.default_rng(0).uniform(0, 1, 5000))
        width = h.edges_[1] - h.edges_[0]
        self.assertAlmostEqual((h.density_ * width).sum(), 1.0, places=6)

    def test_out_of_range_clipped(self):
        h = StreamingHistogram(bins=3, value_range=(0, 1))
        h.update([-5, 5])  
        self.assertEqual(h.counts_.sum(), 2)

    def test_invalid_bins_raises(self):
        with self.assertRaises(ValueError):
            StreamingHistogram(bins=0)


class TestUpdateStats(unittest.TestCase):
    def test_bundle_updates(self):
        rng = np.random.default_rng(0)
        state = None
        for _ in range(5):
            state = update_stats(state, rng.uniform(0, 1, (200, 3)))
        self.assertEqual(state["moments"].count_.tolist(), [1000, 1000, 1000])
        self.assertEqual(len(state["histograms"]), 3)


if __name__ == "__main__":
    unittest.main()

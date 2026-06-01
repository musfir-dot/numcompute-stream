
import unittest

import numpy as np

from numcompute_stream.preprocessing import StandardScaler, Imputer, OneHotEncoder


class TestStandardScaler(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.X = self.rng.normal(10, 4, size=(1000, 3))

    def test_standardises_to_zero_mean_unit_var(self):
        sc = StandardScaler()
        for c in np.array_split(self.X, 10):
            sc.partial_fit(c)
        Z = sc.transform(self.X)
        np.testing.assert_allclose(Z.mean(0), 0, atol=1e-7)
        np.testing.assert_allclose(Z.std(0), 1, atol=1e-2)

    def test_streaming_matches_batch(self):
        stream = StandardScaler()
        for c in np.array_split(self.X, 13):
            stream.partial_fit(c)
        batch = StandardScaler().fit(self.X)
        np.testing.assert_allclose(stream.transform(self.X), batch.transform(self.X), atol=1e-8)

    def test_zero_variance_safe(self):
        X = np.full((50, 2), 5.0)
        sc = StandardScaler().fit(X)
        Z = sc.transform(X)
        self.assertFalse(np.isnan(Z).any())
        self.assertFalse(np.isinf(Z).any())

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            StandardScaler().transform(self.X)

    def test_with_mean_false(self):
        sc = StandardScaler(with_mean=False).fit(self.X)
        Z = sc.transform(self.X)
        self.assertGreater(abs(Z.mean()), 0.1)  # not centred


class TestImputer(unittest.TestCase):
    def test_mean_imputation(self):
        X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
        imp = Imputer("mean").fit(X)
        out = imp.transform(X)
        self.assertFalse(np.isnan(out).any())
        self.assertAlmostEqual(out[1, 1], 4.0)  # mean of 2 and 6

    def test_constant_imputation(self):
        X = np.array([[np.nan, 1.0]])
        out = Imputer("constant", fill_value=-1.0).fit(X).transform(X)
        self.assertEqual(out[0, 0], -1.0)

    def test_median_imputation(self):
        X = np.array([[1.0], [2.0], [3.0], [100.0], [np.nan]])
        out = Imputer("median").fit(X).transform(X)
        self.assertFalse(np.isnan(out).any())

    def test_all_nan_column_uses_fill_value(self):
        X = np.array([[np.nan, 1.0], [np.nan, 2.0]])
        out = Imputer("mean", fill_value=0.0).fit(X).transform(X)
        self.assertEqual(out[0, 0], 0.0)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            Imputer("bogus")


class TestOneHotEncoder(unittest.TestCase):
    def test_basic_encoding(self):
        X = np.array([[0], [1], [2], [1]])
        enc = OneHotEncoder().fit(X)
        out = enc.transform(X)
        self.assertEqual(out.shape, (4, 3))
        np.testing.assert_array_equal(out[0], [1, 0, 0])

    def test_incremental_category_growth(self):
        enc = OneHotEncoder()
        enc.partial_fit(np.array([[0], [1]]))
        self.assertEqual(enc.output_width(), 2)
        enc.partial_fit(np.array([[2], [3]]))  # new categories appear later
        self.assertEqual(enc.output_width(), 4)

    def test_specific_columns(self):
        X = np.array([[0, 9.5], [1, 8.5]])
        enc = OneHotEncoder(columns=[0]).fit(X)
        self.assertEqual(enc.transform(X).shape, (2, 2))

    def test_nan_not_a_category(self):
        X = np.array([[0.0], [np.nan], [1.0]])
        enc = OneHotEncoder().fit(X)
        self.assertEqual(enc.output_width(), 2)


if __name__ == "__main__":
    unittest.main()

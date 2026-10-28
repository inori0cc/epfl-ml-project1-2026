import unittest
import tempfile
from pathlib import Path
import numpy as np


class BoostTests(unittest.TestCase):
    def test_newton_leaf(self):
        from numpy_boost import build_tree

        bins = np.array([[0], [0], [0], [0]], dtype=np.uint8)
        tree = build_tree(
            bins,
            np.array([1.0, 1.0, -1.0, -1.0]),
            np.ones(4),
            np.arange(4),
            [0],
            0,
            2.0,
            1.0,
        )
        self.assertEqual(tree["value"], 0.0)
        tree = build_tree(bins, np.ones(4), np.ones(4), np.arange(4), [0], 0, 2.0, 1.0)
        self.assertAlmostEqual(tree["value"], -4 / 6)

    def test_missing_split_and_roundtrip(self):
        from numpy_boost import HistogramBoost

        rng = np.random.default_rng(4)
        x = rng.normal(size=(400, 3)).astype(np.float32)
        x[:40, 0] = np.nan
        y = (np.nan_to_num(x[:, 0], nan=2) > 0).astype(int)
        m = HistogramBoost(
            rounds=30,
            depth=2,
            learning_rate=0.2,
            bins=16,
            min_child_weight=1,
            colsample=1.0,
            subsample=1.0,
            seed=4,
        )
        m.fit(x[:300], y[:300], ["a", "b", "c"])
        pred = m.predict(x[300:])
        self.assertGreater(((pred > 0.5) == y[300:]).mean(), 0.9)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            m.save(p)
            n = HistogramBoost.load(p)
            np.testing.assert_array_equal(n.predict(x), m.predict(x))
        self.assertTrue(np.isfinite(m.predict(np.full((3, 3), np.nan))).all())

    def test_bins_fitted_only_on_train(self):
        from numpy_boost import fit_cuts, bin_data

        x = np.array([[0.0], [1.0], [2.0], [np.nan]])
        cuts = fit_cuts(x, 8)
        before = [z.copy() for z in cuts]
        b = bin_data(np.array([[1000.0], [np.nan]]), cuts)
        np.testing.assert_array_equal(cuts[0], before[0])
        self.assertEqual(b[1, 0], 0)
        self.assertGreater(b[0, 0], 0)


if __name__ == "__main__":
    unittest.main()

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np


class CoreTests(unittest.TestCase):
    def test_modules_exist(self):
        self.assertIsNotNone(importlib.util.find_spec("implementations"))
        self.assertIsNotNone(importlib.util.find_spec("pipeline"))

    def test_linear_contract(self):
        import implementations as m

        x = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        y = np.array([1.0, 3.0, 5.0])
        w, loss = m.least_squares(y, x)
        self.assertEqual(getattr(loss, "ndim", None), 0)
        np.testing.assert_allclose(w, [1, 2], atol=1e-10)
        self.assertLess(loss, 1e-20)
        w, loss = m.ridge_regression(np.array([2.0, 4.0]), np.eye(2), 0.25)
        np.testing.assert_allclose(w, [1.0, 2.0])
        self.assertAlmostEqual(loss, 1.25)
        initial = np.zeros(2)
        w, loss = m.mean_squared_error_gd(y, x, initial, 1, 0.1)
        np.testing.assert_allclose(w, [0.3, 13 / 30])
        np.testing.assert_array_equal(initial, [0.0, 0.0])
        self.assertAlmostEqual(loss, np.mean((y - x @ w) ** 2) / 2)
        w, loss = m.mean_squared_error_sgd(
            np.array([2.0]), np.array([[1.0]]), np.zeros(1), 1, 0.1
        )
        np.testing.assert_allclose(w, [0.2])
        self.assertAlmostEqual(loss, 1.62)

    def test_logistic_contract(self):
        import implementations as m

        x = np.eye(2)
        y = np.array([0.0, 1.0])
        w, loss = m.logistic_regression(y, x, np.zeros(2), 1, 1.0)
        np.testing.assert_allclose(w, [-0.25, 0.25])
        self.assertAlmostEqual(loss, np.log1p(np.exp(-0.25)))
        w, loss = m.reg_logistic_regression(y, x, 0.1, np.ones(2), 1, 1.0)
        np.testing.assert_allclose(w, [0.4344707106849976, 0.9344707106849976])
        self.assertAlmostEqual(loss, np.mean(np.logaddexp(0.0, x @ w) - y * (x @ w)))
        w, loss = m.logistic_regression(y, x, np.array([-1000.0, 1000.0]), 0, 1.0)
        self.assertTrue(np.isfinite(loss))

    def test_threshold_ties(self):
        from pipeline import best_threshold, metrics

        y = np.array([1, 0, 1, 0])
        p = np.array([0.7, 0.7, 0.4, 0.1])
        t, score = best_threshold(y, p)
        self.assertAlmostEqual(t, 0.4)
        self.assertAlmostEqual(score, 0.8)
        self.assertAlmostEqual(metrics(y, p >= t)["f1"], 0.8)
        self.assertEqual(metrics(y, np.zeros(4))["precision"], 0.0)

    def test_split_disjoint_and_stratified(self):
        from pipeline import stratified_split

        y = np.array([0] * 80 + [1] * 20)
        a, b, c = stratified_split(y, 2026)
        self.assertEqual([len(a), len(b), len(c)], [60, 20, 20])
        self.assertEqual(len(set(a) & set(b) | set(a) & set(c) | set(b) & set(c)), 0)
        self.assertEqual(len(set(a) | set(b) | set(c)), 100)
        for z in [a, b, c]:
            self.assertAlmostEqual(y[z].mean(), 0.2)

    def test_csv_alignment_and_submission(self):
        from pipeline import load_features, load_labels, write_submission

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "x.csv").write_text("Id,a,b\n9,1,\n4,2,3\n")
            ids, x, names = load_features(p / "x.csv")
            np.testing.assert_array_equal(ids, [9, 4])
            self.assertTrue(np.isnan(x[0, 1]))
            self.assertEqual(names, ["a", "b"])
            (p / "y.csv").write_text("Id,_MICHD\n4,-1\n9,1\n")
            np.testing.assert_array_equal(load_labels(p / "y.csv", ids), [1, 0])
            write_submission(ids, np.array([1, -1]), p / "out.csv")
            with (p / "out.csv").open() as f:
                self.assertEqual(
                    list(csv.reader(f)), [["Id", "Prediction"], ["9", "1"], ["4", "-1"]]
                )
            with self.assertRaises(ValueError):
                write_submission(ids, np.array([1]), p / "bad.csv")
            with self.assertRaises(ValueError):
                write_submission(ids, np.array([1, 0]), p / "bad.csv")


if __name__ == "__main__":
    unittest.main()

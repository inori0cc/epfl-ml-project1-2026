import unittest
import numpy as np


class BaseModelTests(unittest.TestCase):
    def test_ridge_entry(self):
        from base_models import fit_model, predict_scores

        w, loss = fit_model("ridge_regression", [2.0, 4.0], np.eye(2), lambda_=0.25)
        np.testing.assert_allclose(w, [1.0, 2.0])
        self.assertAlmostEqual(loss, 1.25)
        np.testing.assert_allclose(
            predict_scores("ridge_regression", np.eye(2), w), [1.0, 2.0]
        )

    def test_all_six_entries(self):
        from base_models import fit_model

        for method in [
            "mean_squared_error_gd",
            "mean_squared_error_sgd",
            "least_squares",
            "ridge_regression",
            "logistic_regression",
            "reg_logistic_regression",
        ]:
            w, loss = fit_model(
                method,
                [0.0, 1.0],
                np.eye(2),
                max_iters=2,
                gamma=0.1,
                lambda_=0.1,
                seed=7,
            )
            self.assertEqual(w.shape, (2,))
            self.assertTrue(np.isfinite(loss))

    def test_logistic_probabilities_and_invalid_input(self):
        from base_models import fit_model, predict_scores

        np.testing.assert_allclose(
            predict_scores("logistic_regression", np.eye(2), np.zeros(2)), [0.5, 0.5]
        )
        with self.assertRaises(ValueError):
            fit_model("unknown", [0, 1], np.eye(2))
        with self.assertRaises(ValueError):
            fit_model("logistic_regression", [-1, 1], np.eye(2))
        with self.assertRaises(ValueError):
            fit_model("ridge_regression", [0, 1], [[np.nan], [1.0]])

    def test_seed_reproducible_without_global_side_effect(self):
        from base_models import fit_model

        np.random.seed(37)
        expected = np.random.random(4)
        np.random.seed(37)
        a = fit_model(
            "mean_squared_error_sgd", [1.0, 3.0], np.eye(2), seed=42, max_iters=12
        )[0]
        np.testing.assert_array_equal(np.random.random(4), expected)
        b = fit_model(
            "mean_squared_error_sgd", [1.0, 3.0], np.eye(2), seed=42, max_iters=12
        )[0]
        np.testing.assert_array_equal(a, b)

    def test_singular_team_least_squares(self):
        import implementations as m

        x = np.array([[1.0, 1.0], [2.0, 2.0]])
        w, loss = m.least_squares(np.array([1.0, 2.0]), x)
        np.testing.assert_allclose(x @ w, [1.0, 2.0], atol=1e-12)
        self.assertLess(loss, 1e-20)


if __name__ == "__main__":
    unittest.main()

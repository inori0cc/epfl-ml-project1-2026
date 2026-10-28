import unittest
import numpy as np


class ViewTests(unittest.TestCase):
    def test_frozen_categories_and_missing(self):
        from feature_views import FeatureView

        x = np.array(
            [[1.0, 0.0], [2.0, 1.0], [np.nan, 2.0], [1.0, 3.0]], dtype=np.float32
        )
        view = FeatureView.fit(x, ["code", "value"], indices=[0], expanded=True)
        z = view.transform(
            np.array([[3.0, 7.0], [np.nan, 8.0], [1.0, 9.0]], dtype=np.float32),
            ["code", "value"],
        )
        self.assertEqual(view.categories, [[1.0, 2.0]])
        np.testing.assert_array_equal(z[0, 1:3], [0.0, 0.0])
        self.assertTrue(np.isnan(z[1, 1:3]).all())
        np.testing.assert_array_equal(z[2, 1:3], [1.0, 0.0])
        np.testing.assert_array_equal(z[:, -1], [0.0, 1.0, 0.0])
        restored = FeatureView.from_dict(view.to_dict())
        np.testing.assert_array_equal(
            restored.transform(x, ["code", "value"]),
            view.transform(x, ["code", "value"]),
        )
        with self.assertRaises(ValueError):
            view.transform(x, ["value", "code"])

    def test_gain_ranking(self):
        from feature_views import rank_features

        trees = [
            dict(
                feature=2,
                gain=3,
                left={"value": 0},
                right=dict(feature=0, gain=4, left={"value": 0}, right={"value": 0}),
            ),
            dict(feature=2, gain=2, left={"value": 0}, right={"value": 0}),
        ]
        self.assertEqual(rank_features(trees, 3, 2), [2, 0])


if __name__ == "__main__":
    unittest.main()

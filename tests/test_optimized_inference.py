import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from numpy_boost import HistogramBoost
from feature_views import FeatureView
from pipeline import sha256
from run_optimized import predict_bundle


class BundleTests(unittest.TestCase):
    def test_weighted_models_roundtrip_and_hash_rejection(self):
        x = np.array([[0.0, 1.0], [1.0, 2.0], [0.0, 3.0], [1.0, 4.0]], dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        names = ["a", "b"]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            members = []
            expected = np.zeros(len(x))
            for i, expanded in enumerate([False, True]):
                view = FeatureView.fit(x, names, expanded=expanded)
                z = view.transform(x, names)
                model = HistogramBoost(
                    rounds=5,
                    depth=1,
                    min_child_weight=0.1,
                    subsample=1.0,
                    colsample=1.0,
                ).fit(z, y, view.names)
                file = root / f"{i}.json"
                model.save(file)
                weight = [0.25, 0.75][i]
                expected += weight * model.predict(z)
                members.append(
                    dict(
                        model=file.name,
                        model_sha256=sha256(file),
                        view=view.to_dict(),
                        weight=weight,
                    )
                )
            meta = dict(input_names=names, members=members, threshold=0.3)
            (root / "metadata.json").write_text(json.dumps(meta))
            actual, threshold = predict_bundle(root, x, names)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(threshold, 0.3)
            with self.assertRaises(ValueError):
                predict_bundle(root, x, names[::-1])
            (root / "0.json").write_text("{}")
            with self.assertRaises(ValueError):
                predict_bundle(root, x, names)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from numpy_boost import HistogramBoost
from pipeline import sha256


class CacheTests(unittest.TestCase):
    def test_changed_model_rejected(self):
        from optimize import load_cached_trial

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            model = p / "model.json"
            record = p / "record.json"
            m = HistogramBoost(rounds=1).fit(
                np.array([[0.0], [1.0]]), np.array([0, 1]), ["a"]
            )
            m.save(model)
            record.write_text(json.dumps({"model_sha256": sha256(model)}))
            rec, restored = load_cached_trial(record, model)
            np.testing.assert_array_equal(
                restored.predict(np.array([[0.0], [1.0]])),
                m.predict(np.array([[0.0], [1.0]])),
            )
            model.write_text("{}")
            with self.assertRaises(ValueError):
                load_cached_trial(record, model)


if __name__ == "__main__":
    unittest.main()

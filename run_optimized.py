"""Reproduce optimization/submission.csv using NumPy only."""

import argparse
import json
from pathlib import Path
import numpy as np
from numpy_boost import HistogramBoost
from feature_views import FeatureView
from pipeline import load_features, sha256, write_submission

ROOT = Path(__file__).resolve().parent


def predict_bundle(model_dir, x, names):
    meta = json.loads((model_dir / "metadata.json").read_text())
    if names != meta["input_names"]:
        raise ValueError("Feature order mismatch")
    prob = np.zeros(len(x))
    for member in meta["members"]:
        path = model_dir / member["model"]
        if sha256(path) != member["model_sha256"]:
            raise ValueError("Model hash mismatch")
        view = FeatureView.from_dict(member["view"])
        model = HistogramBoost.load(path)
        if view.names != model.names:
            raise ValueError("Transformed model schema mismatch")
        prob += member["weight"] * model.predict(view.transform(x, names))
    return prob, meta["threshold"]


def main(default_output=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--model-dir", type=Path, default=ROOT / "optimization/models")
    p.add_argument(
        "--output",
        type=Path,
        default=(
            default_output
            if default_output is not None
            else ROOT / "optimization/submission.csv"
        ),
    )
    a = p.parse_args()
    meta = json.loads((a.model_dir / "metadata.json").read_text())
    if sha256(a.data_dir / "x_test.csv") != meta["data_sha256"]["x_test.csv"]:
        raise ValueError("Data hash mismatch")
    ids, x, names = load_features(a.data_dir / "x_test.csv")
    prob, t = predict_bundle(a.model_dir, x, names)
    write_submission(ids, np.where(prob >= t, 1, -1), a.output)
    print(json.dumps(dict(rows=len(ids), sha256=sha256(a.output))))


if __name__ == "__main__":
    main()

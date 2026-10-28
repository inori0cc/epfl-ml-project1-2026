"""Recreate the submission from the saved final model; no retraining."""

import argparse
import json
from pathlib import Path
import numpy as np
from numpy_boost import HistogramBoost
from pipeline import load_features, sha256, write_submission

ROOT = Path(__file__).resolve().parent


def predict_model(model_path, x, feature_names):
    model = HistogramBoost.load(model_path)
    if model.names != list(feature_names):
        raise ValueError("Feature schema/order differs from saved model")
    return model.predict(x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--model-dir", type=Path, default=ROOT / "models")
    p.add_argument("--output", type=Path, default=ROOT / "submission_baseline.csv")
    args = p.parse_args()
    meta = json.loads((args.model_dir / "metadata.json").read_text())
    test_path = args.data_dir / "x_test.csv"
    if sha256(test_path) != meta["data_sha256"]["x_test.csv"]:
        raise ValueError("Test data differs from the recorded official dataset")
    ids, x, names = load_features(test_path)
    if names != meta["feature_names"]:
        raise ValueError("Feature schema differs from training")
    model_path = args.model_dir / "final.json"
    if sha256(model_path) != meta["model_sha256"]:
        raise ValueError("Model hash mismatch")
    prob = predict_model(model_path, x, names)
    labels = np.where(prob >= meta["threshold"], 1, -1)
    write_submission(ids, labels, args.output)
    print(
        json.dumps(
            dict(
                rows=len(ids),
                positive=int((labels == 1).sum()),
                sha256=sha256(args.output),
            )
        )
    )


if __name__ == "__main__":
    main()

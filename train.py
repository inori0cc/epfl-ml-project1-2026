"""Refit the two selected ensemble members on the official labeled data."""

import argparse
import gc
import json
from pathlib import Path
import numpy as np
from feature_views import FeatureView
from numpy_boost import HistogramBoost
from pipeline import load_features, load_labels, sha256, write_submission

ROOT = Path(__file__).resolve().parent


def main():
    """Reuse the frozen configuration and threshold; write to a fresh directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "retrained")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Use a new output directory to preserve existing models")
    meta = json.loads((ROOT / "models/metadata.json").read_text())
    for name, digest in meta["data_sha256"].items():
        if sha256(args.data_dir / name) != digest:
            raise ValueError("Official data hash mismatch: " + name)
    ids, x, names = load_features(args.data_dir / "x_train.csv")
    y = load_labels(args.data_dir / "y_train.csv", ids)
    test_ids, xt, test_names = load_features(args.data_dir / "x_test.csv")
    if names != meta["input_names"] or test_names != names:
        raise ValueError("Feature schema mismatch")
    args.output_dir.mkdir(parents=True)
    probabilities = np.zeros(len(xt))
    for member in meta["members"]:
        original = ROOT / "models" / member["model"]
        if sha256(original) != member["model_sha256"]:
            raise ValueError("Reference model hash mismatch")
        saved = HistogramBoost.load(original)
        params = dict(saved.params)
        params["rounds"] = len(saved.trees)
        view = FeatureView.from_dict(member["view"])
        print("Training " + member["model"], flush=True)
        model = HistogramBoost(**params).fit(view.transform(x, names), y, view.names)
        destination = args.output_dir / member["model"]
        model.save(destination)
        member["model_sha256"] = sha256(destination)
        probabilities += member["weight"] * model.predict(view.transform(xt, names))
        del model, saved
        gc.collect()
    output = args.output_dir / "submission.csv"
    write_submission(
        test_ids, np.where(probabilities >= meta["threshold"], 1, -1), output
    )
    meta["submission_sha256"] = sha256(output)
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(dict(rows=len(test_ids), sha256=sha256(output))))


if __name__ == "__main__":
    main()

"""Train baselines and NumPy boosting, freeze selection, then evaluate holdout once.

Training and inference require only NumPy and Python standard libraries.
All comparisons and thresholds are selected on tune only, never holdout.
"""

import argparse
import csv
import gc
import json
import platform
import time
import warnings
from pathlib import Path
import numpy as np
from numpy_boost import HistogramBoost
from base_models import fit_model, predict_scores
from pipeline import (
    best_threshold,
    load_features,
    load_labels,
    metrics,
    sha256,
    stratified_split,
    write_submission,
)

ROOT = Path(__file__).resolve().parent


def ridge_features(x, means=None, scales=None):
    """Mean imputation, standardization and intercept, fitted on train only."""
    if means is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(x, axis=0, dtype=np.float64)
        means = np.nan_to_num(means)
    z = np.where(np.isnan(x), means, x).astype(np.float64)
    if scales is None:
        scales = z.std(axis=0)
        scales[scales < 1e-8] = 1.0
    z = (z - means) / scales
    return np.column_stack([np.ones(len(z)), z]), means, scales


def bootstrap_f1(y, pred, seed=7, n=1000):
    """Percentile interval via multinomial resampling of confusion categories."""
    m = metrics(y, pred)
    counts = np.array([m["tp"], m["fp"], m["fn"], m["tn"]])
    draws = np.random.default_rng(seed).multinomial(len(y), counts / len(y), size=n)
    f1 = 2 * draws[:, 0] / np.maximum(2 * draws[:, 0] + draws[:, 1] + draws[:, 2], 1)
    return [float(v) for v in np.quantile(f1, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-rounds", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    args = parser.parse_args()
    out = args.output_dir
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("Loading official CSV data", flush=True)
    ids, x, names = load_features(args.data_dir / "x_train.csv")
    y = load_labels(args.data_dir / "y_train.csv", ids)
    test_ids, xt, test_names = load_features(args.data_dir / "x_test.csv")
    if names != test_names or set(ids) & set(test_ids):
        raise ValueError("Schema mismatch or overlapping train/test IDs")
    hashes = {
        n: sha256(args.data_dir / n)
        for n in ["x_train.csv", "y_train.csv", "x_test.csv"]
    }
    a, b, c = stratified_split(y, args.seed)
    np.savez_compressed(
        out / "results" / "split_ids.npz", train=ids[a], tune=ids[b], holdout=ids[c]
    )
    eda = dict(
        train_rows=len(y),
        test_rows=len(xt),
        features=len(names),
        positive=int(y.sum()),
        positive_fraction=float(y.mean()),
        missing_fraction=float(np.isnan(x).mean()),
        split_sizes=[len(a), len(b), len(c)],
        split_positive=[int(y[z].sum()) for z in [a, b, c]],
    )
    with (out / "results" / "feature_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "missing_fraction", "unique_nonmissing"])
        for j, n in enumerate(names):
            w.writerow(
                [
                    n,
                    float(np.isnan(x[:, j]).mean()),
                    len(np.unique(x[~np.isnan(x[:, j]), j])),
                ]
            )
    print(json.dumps(eda), flush=True)
    records = []
    predictions = {}
    records.append(
        dict(
            name="always_negative",
            threshold=0.5,
            tune=metrics(y[b], np.zeros(len(b))),
            holdout=metrics(y[c], np.zeros(len(c))),
        )
    )
    # Ridge hyperparameter selection uses tune only; holdout never changes choices.
    print("Training ridge baselines", flush=True)
    za, means, scales = ridge_features(x[a])
    zb, _, _ = ridge_features(x[b], means, scales)
    ridge_trials = []
    for lam in [1e-4, 1e-2, 1.0]:
        w, _ = fit_model("ridge_regression", y[a], za, lambda_=lam)
        prob = predict_scores("ridge_regression", zb, w)
        threshold, _ = best_threshold(y[b], prob)
        rec = dict(
            name="ridge",
            lambda_=lam,
            threshold=threshold,
            tune=metrics(y[b], prob >= threshold),
        )
        ridge_trials.append((rec, w, prob))
    ridge_rec, ridge_w, ridge_tune = max(ridge_trials, key=lambda z: z[0]["tune"]["f1"])
    del za, zb
    gc.collect()
    zc, _, _ = ridge_features(x[c], means, scales)
    ridge_hold = predict_scores("ridge_regression", zc, ridge_w)
    del zc
    gc.collect()
    ridge_rec["holdout"] = metrics(y[c], ridge_hold >= ridge_rec["threshold"])
    records.append(ridge_rec)
    predictions["ridge_tune"] = ridge_tune
    predictions["ridge_holdout"] = ridge_hold
    configs = [
        ("numpy_depth2", {"depth": 2}),
        ("numpy_depth3", {"depth": 3}),
        ("numpy_depth3_weighted", {"depth": 3, "pos_weight": 3.0}),
    ]
    chosen = None
    # Cache candidate tune scores and sliced models. Do not inspect holdout here.
    for name, overrides in configs:
        tick = time.time()
        params = dict(rounds=args.max_rounds, seed=args.seed) | overrides
        print("Training " + name, flush=True)
        model = HistogramBoost(**params).fit(
            x[a], y[a], names, validation=(x[b], y[b]), patience=args.patience
        )
        rounds = len(model.trees)
        prob = model.predict(x[b])
        threshold, _ = best_threshold(y[b], prob)
        rec = dict(
            name=name,
            params=params,
            rounds=rounds,
            threshold=threshold,
            tune=metrics(y[b], prob >= threshold),
            tune_at_05=metrics(y[b], prob >= 0.5),
            seconds=time.time() - tick,
        )
        model.save(out / "models" / (name + ".json"))
        predictions[name + "_tune"] = prob
        records.append(rec)
        if chosen is None or rec["tune"]["f1"] > chosen["tune"]["f1"]:
            chosen = rec
        print(json.dumps(rec), flush=True)
        del model
        gc.collect()
    # Selection frozen before any NumPy boosting holdout prediction or metric is computed.
    (out / "results" / "selection.json").write_text(json.dumps(chosen, indent=2) + "\n")
    print("Frozen choice: " + chosen["name"], flush=True)
    dc = x[c]
    for rec in records:
        if not rec["name"].startswith("numpy_"):
            continue
        model = HistogramBoost.load(out / "models" / (rec["name"] + ".json"))
        prob = model.predict(dc)
        rec["holdout"] = metrics(y[c], prob >= rec["threshold"])
        rec["holdout_at_05"] = metrics(y[c], prob >= 0.5)
        predictions[rec["name"] + "_holdout"] = prob
        if rec["name"] == chosen["name"]:
            rec["holdout_f1_95ci"] = bootstrap_f1(y[c], prob >= rec["threshold"])
    del dc, model
    gc.collect()
    np.savez_compressed(
        out / "results" / "validation_predictions.npz",
        y_tune=y[b],
        y_holdout=y[c],
        **predictions
    )
    print("Refitting " + chosen["name"] + " on all labeled rows", flush=True)
    final_params = chosen["params"] | {"rounds": chosen["rounds"]}
    final = HistogramBoost(**final_params).fit(x, y, names)
    final.save(out / "models" / "final.json")
    prob = final.predict(xt)
    write_submission(
        test_ids, np.where(prob >= chosen["threshold"], 1, -1), out / "submission.csv"
    )
    np.save(out / "results" / "test_probabilities.npy", prob)
    meta = dict(
        feature_names=names,
        threshold=chosen["threshold"],
        params=chosen["params"],
        rounds=chosen["rounds"],
        selected=chosen["name"],
        seed=args.seed,
        data_sha256=hashes,
        model_sha256=sha256(out / "models" / "final.json"),
        submission_sha256=sha256(out / "submission.csv"),
        python=platform.python_version(),
        numpy=np.__version__,
        algorithm="original NumPy histogram Newton boosting",
    )
    (out / "models" / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    results = dict(
        eda=eda,
        experiments=records,
        ridge_search=[v[0] for v in ridge_trials],
        selected=chosen["name"],
        max_rounds=args.max_rounds,
        patience=args.patience,
        total_seconds=time.time() - started,
        submission_rows=len(test_ids),
        submission_positive=int((prob >= chosen["threshold"]).sum()),
        ai_crowd_score=None,
        library_restriction_compliant=True,
    )
    (out / "results" / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

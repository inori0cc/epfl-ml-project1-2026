"""Data validation, deterministic splits, evaluation and submission utilities."""

import csv
import hashlib
from pathlib import Path
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def load_features(path):
    """Load a numeric CSV, storing small official IDs as int64 and features as float32.

    This loader is for the official dataset; it parses through float64 and is
    not intended for arbitrary identifiers larger than 2**53.
    """
    path = Path(path)
    with path.open() as f:
        header = next(csv.reader(f))
    names = header[1:]
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty or duplicate feature names")
    # Empty official CSV fields and literal NaN both represent missing values.
    raw = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        dtype=np.float64,
        ndmin=2,
        converters=lambda s: float(s) if s.strip() else np.nan,
    )
    if raw.shape[1] != len(header):
        raise ValueError("Feature/header width mismatch")
    if (
        not np.isfinite(raw[:, 0]).all()
        or not np.equal(raw[:, 0], np.floor(raw[:, 0])).all()
    ):
        raise ValueError("IDs must be finite integers")
    ids = raw[:, 0].astype(np.int64)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs")
    x = raw[:, 1:].astype(np.float32)
    if np.isinf(x).any():
        raise ValueError("Infinite feature value")
    return ids, x, names


def load_labels(path, ids):
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if data.shape != (len(ids), 2) or not np.isin(data[:, 1], [-1, 1]).all():
        raise ValueError("Label shape or values invalid")
    label_ids = data[:, 0].astype(np.int64)
    if (
        not np.equal(label_ids, data[:, 0]).all()
        or len(set(label_ids)) != len(ids)
        or set(label_ids) != set(ids)
    ):
        raise ValueError("Label and feature IDs do not match")
    order = np.argsort(label_ids)
    aligned = data[order[np.searchsorted(label_ids[order], ids)], 1]
    return (aligned == 1).astype(np.int8)


def stratified_split(y, seed):
    """Return mutually exclusive 60/20/20 train/tune/holdout indices."""
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for c in [0, 1]:
        ix = rng.permutation(np.flatnonzero(y == c))
        a, b = int(0.6 * len(ix)), int(0.8 * len(ix))
        for k, z in enumerate([ix[:a], ix[a:b], ix[b:]]):
            parts[k].extend(z)
    return tuple(rng.permutation(np.array(z, dtype=np.int64)) for z in parts)


def metrics(y, pred):
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    if y.shape != pred.shape or y.ndim != 1 or not len(y):
        raise ValueError("Expected equal nonempty 1D labels/predictions")
    tp = int(np.sum(y & pred))
    fp = int(np.sum(~y & pred))
    fn = int(np.sum(y & ~pred))
    tn = int(np.sum(~y & ~pred))
    return dict(
        f1=2 * tp / max(2 * tp + fp + fn, 1),
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        accuracy=(tp + tn) / len(y),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def best_threshold(y, scores):
    """Exact maximum F1 over score cutoffs, with ties kept together."""
    y, scores = np.asarray(y), np.asarray(scores)
    if (
        y.shape != scores.shape
        or y.ndim != 1
        or not len(y)
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Invalid threshold inputs")
    order = np.argsort(-scores, kind="stable")
    s, labels = scores[order], y[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    tp = np.cumsum(labels)[ends]
    f1 = 2 * tp / np.maximum(ends + 1 + y.sum(), 1)
    j = int(np.argmax(f1))
    return float(s[ends[j]]), float(f1[j])


def write_submission(ids, pred, path):
    ids, pred = np.asarray(ids), np.asarray(pred)
    if ids.ndim != 1 or pred.shape != ids.shape or not np.isin(pred, [-1, 1]).all():
        raise ValueError("Submission requires matching 1D IDs and {-1,1} labels")
    if len(set(ids)) != len(ids) or not np.equal(ids, np.floor(ids)).all():
        raise ValueError("Submission IDs must be unique integers")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Prediction"])
        writer.writerows((int(i), int(p)) for i, p in zip(ids, pred))

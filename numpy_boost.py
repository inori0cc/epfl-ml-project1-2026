"""Original NumPy implementation of histogram Newton gradient boosting.

This implements the regularized second-order tree objective, not the XGBoost
library. Bin 0 is missing; each split tests missing-left and missing-right.
No compiled ML library, copied tree code, or pretrained model is used.
"""

import json
from pathlib import Path
import numpy as np
from pipeline import best_threshold


def sigmoid(z):
    """Stable boosting probability link, retained from the fitted model version."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(z)))


def fit_cuts(x, bins):
    if not 2 <= bins <= 254:
        raise ValueError("bins must be between 2 and 254")
    cuts = []
    for j in range(x.shape[1]):
        v = x[np.isfinite(x[:, j]), j]
        if not len(v):
            cuts.append(np.array([], dtype=float))
            continue
        u = np.unique(v)
        if len(u) <= bins:
            cut = (u[:-1].astype(float) + u[1:].astype(float)) / 2
        else:
            cut = np.unique(np.quantile(v, np.arange(1, bins) / bins))
        cuts.append(cut)
    return cuts


def bin_data(x, cuts):
    if x.ndim != 2 or x.shape[1] != len(cuts):
        raise ValueError("Feature width mismatch")
    b = np.empty(x.shape, dtype=np.uint8, order="F")
    for j, cut in enumerate(cuts):
        b[:, j] = np.searchsorted(cut, x[:, j], side="right") + 1
        b[np.isnan(x[:, j]), j] = 0
    return b


def build_tree(b, g, h, rows, features, depth, reg_lambda, min_child_weight):
    """Greedy split gain: .5*(GL²/(HL+l)+GR²/(HR+l)-G²/(H+l))."""
    G = float(g[rows].sum())
    H = float(h[rows].sum())
    leaf = {"value": -G / (H + reg_lambda)}
    if depth == 0 or H < 2 * min_child_weight or len(rows) < 2:
        return leaf
    gr = g[rows]
    hr = h[rows]
    best_gain = 0.0
    best = None
    for j in features:
        codes = b[rows, j]
        gh = np.bincount(codes, weights=gr, minlength=256)
        hh = np.bincount(codes, weights=hr, minlength=256)
        # Include boundary 0: a pure missing/nonmissing split is permitted.
        cg = np.r_[0.0, np.cumsum(gh[1:255])]
        ch = np.r_[0.0, np.cumsum(hh[1:255])]
        for missing_left in [False, True]:
            gl = cg + (gh[0] if missing_left else 0.0)
            hl = ch + (hh[0] if missing_left else 0.0)
            rh = H - hl
            valid = (hl >= min_child_weight) & (rh >= min_child_weight)
            gain = np.full(len(gl), -np.inf)
            gain[valid] = 0.5 * (
                gl[valid] ** 2 / (hl[valid] + reg_lambda)
                + (G - gl[valid]) ** 2 / (rh[valid] + reg_lambda)
                - G**2 / (H + reg_lambda)
            )
            k = int(np.argmax(gain))
            if gain[k] > best_gain + 1e-12:
                best_gain = float(gain[k])
                best = (int(j), k, missing_left)
    if best is None:
        return leaf
    j, k, missing_left = best
    codes = b[rows, j]
    left = ((codes > 0) & (codes <= k)) | ((codes == 0) & missing_left)
    if not left.any() or left.all():
        return leaf
    return dict(
        feature=j,
        cut=k,
        missing_left=missing_left,
        gain=best_gain,
        left=build_tree(
            b, g, h, rows[left], features, depth - 1, reg_lambda, min_child_weight
        ),
        right=build_tree(
            b, g, h, rows[~left], features, depth - 1, reg_lambda, min_child_weight
        ),
    )


def tree_predict(tree, b):
    out = np.empty(len(b), dtype=float)

    def visit(node, rows):
        if "value" in node:
            out[rows] = node["value"]
            return
        v = b[rows, node["feature"]]
        mask = ((v > 0) & (v <= node["cut"])) | ((v == 0) & node["missing_left"])
        visit(node["left"], rows[mask])
        visit(node["right"], rows[~mask])

    visit(tree, np.arange(len(b)))
    return out


class HistogramBoost:
    def __init__(
        self,
        rounds=300,
        depth=3,
        learning_rate=0.1,
        bins=32,
        min_child_weight=20.0,
        reg_lambda=10.0,
        colsample=0.35,
        subsample=0.7,
        pos_weight=1.0,
        seed=2026,
    ):
        self.params = dict(
            rounds=rounds,
            depth=depth,
            learning_rate=learning_rate,
            bins=bins,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            colsample=colsample,
            subsample=subsample,
            pos_weight=pos_weight,
            seed=seed,
        )

    def fit(self, x, y, names, validation=None, patience=60):
        p = self.params
        y = np.asarray(y, dtype=float)
        if not np.isin(y, [0, 1]).all() or len(y) != len(x):
            raise ValueError("Invalid labels")
        self.names = list(names)
        self.cuts = fit_cuts(x, p["bins"])
        b = bin_data(x, self.cuts)
        weights = np.where(y == 1, p["pos_weight"], 1.0)
        prevalence = np.clip(np.average(y, weights=weights), 1e-6, 1 - 1e-6)
        self.base = float(np.log(prevalence / (1 - prevalence)))
        self.trees = []
        self.history = []
        score = np.full(len(y), self.base)
        rng = np.random.default_rng(p["seed"])
        if validation is not None:
            xv, yv = validation
            bv = bin_data(xv, self.cuts)
            sv = np.full(len(yv), self.base)
        best = -np.inf
        best_round = 0
        for i in range(p["rounds"]):
            prob = sigmoid(score)
            g = weights * (prob - y)
            h = weights * np.maximum(prob * (1 - prob), 1e-8)
            rows = rng.choice(
                len(y), max(1, int(len(y) * p["subsample"])), replace=False
            )
            features = np.sort(
                rng.choice(
                    x.shape[1], max(1, int(x.shape[1] * p["colsample"])), replace=False
                )
            )
            tree = build_tree(
                b,
                g,
                h,
                rows,
                features,
                p["depth"],
                p["reg_lambda"],
                p["min_child_weight"],
            )
            self.trees.append(tree)
            score += p["learning_rate"] * tree_predict(tree, b)
            if validation is not None:
                sv += p["learning_rate"] * tree_predict(tree, bv)
                if (i + 1) % 10 == 0 or i == p["rounds"] - 1:
                    threshold, f1 = best_threshold(yv, sigmoid(sv))
                    self.history.append(dict(round=i + 1, f1=f1, threshold=threshold))
                    print(f"round={i+1} tune_f1={f1:.6f}", flush=True)
                    if f1 > best + 1e-6:
                        best = f1
                        best_round = i + 1
                    if i + 1 - best_round >= patience:
                        break
        if validation is not None:
            self.trees = self.trees[:best_round]
        return self

    def predict(self, x):
        b = bin_data(x, self.cuts)
        score = np.full(len(x), self.base)
        for tree in self.trees:
            score += self.params["learning_rate"] * tree_predict(tree, b)
        return sigmoid(score)

    def save(self, path):
        Path(path).write_text(
            json.dumps(
                dict(
                    params=self.params,
                    names=self.names,
                    cuts=[v.tolist() for v in self.cuts],
                    base=self.base,
                    trees=self.trees,
                    history=self.history,
                )
            )
            + "\n"
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        m = cls(**data["params"])
        m.names = data["names"]
        m.cuts = [np.array(v) for v in data["cuts"]]
        m.base = data["base"]
        m.trees = data["trees"]
        m.history = data["history"]
        return m

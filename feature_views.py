"""Train-fitted feature selection and categorical expansion; NumPy only."""

import numpy as np


def rank_features(trees, n_features, limit):
    gain = np.zeros(n_features)

    def visit(t):
        if "value" in t:
            return
        gain[t["feature"]] += t["gain"]
        visit(t["left"])
        visit(t["right"])

    for tree in trees:
        visit(tree)
    return np.argsort(-gain, kind="stable")[: min(limit, n_features)].tolist()


class FeatureView:
    @classmethod
    def fit(cls, x, names, indices=None, expanded=False):
        m = cls()
        m.input_names = list(names)
        m.indices = list(range(len(names))) if indices is None else list(indices)
        m.expanded = expanded
        m.categories = []
        m.names = [names[i] for i in m.indices]
        for i in m.indices:
            u = np.unique(x[np.isfinite(x[:, i]), i]) if expanded else []
            values = [float(v) for v in u] if 2 <= len(u) <= 12 else []
            m.categories.append(values)
            m.names.extend(f"{names[i]}__eq_{v:g}" for v in values)
        if expanded:
            m.names.append("__selected_missing_fraction")
        return m

    def transform(self, x, names):
        if list(names) != self.input_names or x.shape[1] != len(names):
            raise ValueError("Feature schema mismatch")
        selected = x[:, self.indices]
        columns = [selected]
        if self.expanded:
            for j, values in enumerate(self.categories):
                for value in values:
                    z = (selected[:, j] == value).astype(np.float32)
                    z[np.isnan(selected[:, j])] = np.nan
                    columns.append(z[:, None])
            columns.append(np.isnan(selected).mean(axis=1, dtype=np.float32)[:, None])
        return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)

    def to_dict(self):
        return dict(
            input_names=self.input_names,
            indices=self.indices,
            expanded=self.expanded,
            categories=self.categories,
            names=self.names,
        )

    @classmethod
    def from_dict(cls, data):
        m = cls()
        for k, v in data.items():
            setattr(m, k, v)
        return m

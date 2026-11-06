"""Gradient-boosted verifier stored as plain JSON and evaluated with numpy only (no scikit-learn at run time).
Used by detector.Verifier.from_json when the json has "type": "gbt"."""
import numpy as np


class GBTVerifier:
    def __init__(self, d):
        self.mean, self.std = np.asarray(d["mean"], np.float64), np.asarray(d["std"], np.float64)
        self.baseline = float(d["baseline"])
        self.trees = [(np.asarray(t["f"]), np.asarray(t["t"], np.float64), np.asarray(t["l"]), np.asarray(t["r"]),
                       np.asarray(t["leaf"], bool), np.asarray(t["v"], np.float64), np.asarray(t["ml"], bool)) for t in d["trees"]]
        self.threshold = float(d["threshold"])
        self.unit_gate = d.get("unit_gate")
        self.idx = None
        self._d = d

    def score(self, feats) -> float:
        z = (np.asarray(feats, np.float64) - self.mean) / self.std
        raw = self.baseline
        for f, t, l, r, leaf, v, ml in self.trees:
            i = 0
            while not leaf[i]:
                x = z[f[i]]
                i = l[i] if (ml[i] if np.isnan(x) else x <= t[i]) else r[i]
            raw += v[i]
        return float(1.0 / (1.0 + np.exp(-raw)))

    def to_json(self):
        d = dict(self._d); d["threshold"] = self.threshold; d["unit_gate"] = self.unit_gate
        return d

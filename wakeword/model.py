"""Frame-level acoustic model (small MLP over stacked log-mel context).

Training uses scikit-learn; inference is plain numpy (weights are stored in a .npz), so the
running system does not depend on the scikit-learn version that trained the model.
"""
from __future__ import annotations

import csv
import warnings
import json
import time
from pathlib import Path

import numpy as np

from . import audio_io
from .config import BG, CLASSES, DATA_DIR, GLE, GOO, HEY, MODEL_DIR, OTHER
from .disfluency import segments_to_frame_labels
from .features import frame_features


class FrameModel:
    def __init__(self, weights, biases, mean, std, priors=None):
        self.W = [np.asarray(w, dtype=np.float32) for w in weights]
        self.b = [np.asarray(b, dtype=np.float32) for b in biases]
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.priors = np.asarray(priors if priors is not None else np.ones(len(CLASSES)) / len(CLASSES), dtype=np.float32)

    # ---- inference -------------------------------------------------------------
    def logits(self, feats: np.ndarray) -> np.ndarray:
        h = (feats - self.mean) / self.std
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.maximum(h @ W + b, 0.0)
        return h @ self.W[-1] + self.b[-1]

    def log_posteriors(self, feats: np.ndarray) -> np.ndarray:
        z = self.logits(feats)
        z = z - z.max(axis=1, keepdims=True)
        return (z - np.log(np.exp(z).sum(axis=1, keepdims=True))).astype(np.float32)

    def predict_wave(self, x: np.ndarray) -> np.ndarray:
        f = frame_features(x)
        if f.shape[0] == 0:
            return np.zeros((0, len(CLASSES)), dtype=np.float32)
        return self.log_posteriors(f)

    # ---- persistence -------------------------------------------------------------
    def save(self, path):
        data = {"mean": self.mean, "std": self.std, "priors": self.priors, "n_layers": np.array(len(self.W))}
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            data[f"W{i}"], data[f"b{i}"] = W, b
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        n = int(d["n_layers"])
        return cls([d[f"W{i}"] for i in range(n)], [d[f"b{i}"] for i in range(n)], d["mean"], d["std"], d["priors"])


# ----------------------------------------------------------------------------- data access
def read_manifest(data_dir=DATA_DIR):
    with open(Path(data_dir) / "manifest.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_annotations(data_dir=DATA_DIR):
    out = {}
    with open(Path(data_dir) / "annotations.jsonl", encoding="utf-8") as fh:
        for line in fh:
            a = json.loads(line)
            out[a["sample_id"]] = a
    return out


def load_sample(data_dir, row, ann):
    x = audio_io.load_audio(Path(data_dir) / row["relative_path"])
    lab = segments_to_frame_labels([tuple(s) for s in ann[row["sample_id"]]["segments"]], len(x))
    return x, lab


def collect_frames(data_dir, rows, ann, rng, p_bg=0.3, p_other=0.55, log=print):
    """Frame features/labels of the given clips.

    Memory-friendly: keyword frames (hey / goo / gle) are always kept, background and other-speech
    frames are subsampled clip by clip, so the full frame matrix is never materialised.
    """
    Xs, ys = [], []
    for i, row in enumerate(rows):
        x, lab = load_sample(data_dir, row, ann)
        f = frame_features(x)
        n = min(len(f), len(lab))
        f, lab = f[:n], lab[:n]
        u = rng.random(n)
        keep = np.isin(lab, [HEY, GOO, GLE]) | ((lab == BG) & (u < p_bg)) | ((lab == OTHER) & (u < p_other))
        Xs.append(f[keep].astype(np.float16))
        ys.append(lab[keep])
        if (i + 1) % 500 == 0:
            log(f"      features {i + 1}/{len(rows)}")
    return np.concatenate(Xs), np.concatenate(ys)


# ----------------------------------------------------------------------------- training
def train_frame_model(data_dir=DATA_DIR, out_path=None, epochs: int = 14, hidden=(256, 128), seed: int = 0,
                      log=print, max_train_clips=None):
    from sklearn.neural_network import MLPClassifier
    warnings.filterwarnings("ignore", message="Got `batch_size`")

    rng = np.random.default_rng(seed)
    rows = read_manifest(data_dir)
    ann = read_annotations(data_dir)
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    if max_train_clips:
        tr = [tr[i] for i in rng.permutation(len(tr))[:max_train_clips]]
    log(f"[train] {len(tr)} train clips, {len(va)} val clips")
    t0 = time.time()
    Xtr, ytr = collect_frames(data_dir, tr, ann, rng, log=log)
    Xva, yva = collect_frames(data_dir, va, ann, rng, log=log)
    log(f"[train] frames  train={len(Xtr)} val={len(Xva)}  ({time.time() - t0:.0f}s)  "
        f"class counts={dict(zip(CLASSES, np.bincount(ytr, minlength=5).tolist()))}")
    s1 = np.zeros(Xtr.shape[1], dtype=np.float64)
    s2 = np.zeros(Xtr.shape[1], dtype=np.float64)
    for i in range(0, len(Xtr), 50000):
        c = Xtr[i:i + 50000].astype(np.float64)
        s1 += c.sum(axis=0)
        s2 += (c ** 2).sum(axis=0)
    mean = (s1 / len(Xtr)).astype(np.float32)
    std = (np.sqrt(np.maximum(s2 / len(Xtr) - mean.astype(np.float64) ** 2, 0)) + 1e-3).astype(np.float32)
    norm = lambda a: (a.astype(np.float32) - mean) / std

    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=1e-3,
                        learning_rate_init=2e-3, batch_size=512, random_state=seed)
    classes = np.arange(len(CLASSES))
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(perm), 512):
            b = np.sort(perm[s:s + 512])
            clf.partial_fit(norm(Xtr[b]), ytr[b], classes=classes)
        pred = np.concatenate([clf.predict(norm(Xva[i:i + 50000])) for i in range(0, len(Xva), 50000)])
        acc = float((pred == yva).mean())
        # macro recall over classes (robust to the class balance)
        rec = np.mean([(pred[yva == c] == c).mean() for c in classes if (yva == c).any()])
        hist.append({"epoch": ep + 1, "val_acc": round(acc, 4), "val_macro_recall": round(float(rec), 4)})
        log(f"[train] epoch {ep + 1:2d}/{epochs}  val acc={acc:.3f}  macro-recall={rec:.3f}  ({time.time() - t0:.0f}s)")
        if rec > best:
            best = rec
            best_state = ([w.copy() for w in clf.coefs_], [b.copy() for b in clf.intercepts_])
        if ep == int(epochs * 0.6) and getattr(clf, "_optimizer", None) is not None:
            clf._optimizer.learning_rate_init = 5e-4          # decay the learning rate for the last epochs
    W, b = best_state
    priors = np.bincount(ytr, minlength=5) / len(ytr)
    model = FrameModel(W, b, mean, std, priors)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(out_path)
    return model, hist

"""High-level wake-word detector: audio in -> decision + disfluency analysis + fluent output."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import decoder, dsp
from .config import CLASSES, HOP, KEYWORD_TEXT, MODEL_DIR, OTHER, SR, WIN
from .features import frame_db
from .model import FrameModel

UNIT_TEXT = {"H": "Hey", "G": "Goo", "L": "gle"}
PROLONG_S = {"H": 0.55, "G": 0.55}          # a unit held longer than this is called a prolongation
LEAD_FRAMES, LEAD_OTHER_MAX = 25, 0.6       # live listening: at most 60 % of the 0.25 s before the phrase may be speech
BLOCK_S = 0.30                              # silence longer than this inside the phrase = block


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class Verifier:
    """Small feed-forward net over the decoder evidence (logistic regression = one layer).

    Stored as json and evaluated in numpy, so it needs no ML library at run time."""

    def __init__(self, mean, std, layers, threshold, idx=None, unit_gate=None):
        self.unit_gate = unit_gate          # chosen on validation data (None = config.UNIT_GATE)
        # idx = which decoder features the verifier reads (None = all of them, the old format)
        self.idx = None if idx is None else np.asarray(idx, dtype=np.int64)
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.layers = [(np.asarray(W, np.float32), np.asarray(b, np.float32)) for W, b in layers]
        self.threshold = float(threshold)

    def score(self, feats) -> float:
        f = np.asarray(feats, np.float32)
        if self.idx is not None:
            f = f[self.idx]
        h = (f.astype(np.float64) - self.mean) / self.std
        for W, b in self.layers[:-1]:
            h = np.maximum(h @ W + b, 0.0)
        W, b = self.layers[-1]
        return float(_sigmoid(float((h @ W + b).reshape(-1)[0])))

    def to_json(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in self.layers],
                "threshold": self.threshold,
                "features": [decoder.FEATURE_NAMES[i] for i in self.idx] if self.idx is not None else decoder.FEATURE_NAMES,
                "idx": None if self.idx is None else self.idx.tolist(), "unit_gate": self.unit_gate}

    @classmethod
    def from_json(cls, d):
        if d.get("type") == "gbt":                     # v1.7: gradient-boosted verifier (pure numpy, see gbt_verifier.py)
            from .gbt_verifier import GBTVerifier
            return GBTVerifier(d)
        if "layers" in d:
            layers = [(l["W"], l["b"]) for l in d["layers"]]
        else:                                          # old logistic-regression format
            layers = [(np.asarray(d["w"])[:, None], [d["b"]])]
        return cls(d["mean"], d["std"], layers, d["threshold"], d.get("idx"), d.get("unit_gate"))


def _heard_text(units):
    """Rough orthographic rendering of what was *said*, e.g.  H-H-Hey … Goo-Goooogle  (for the UI)."""
    idx_h = [i for i, u in enumerate(units) if u[0] == "H"]
    idx_g = [i for i, u in enumerate(units) if u[0] == "G"]
    last_h = idx_h[-1] if idx_h else -1
    last_g = idx_g[-1] if idx_g else -1
    out, seen_g = "", False
    for i, (k, s, e, d) in enumerate(units):
        if k in ("HG", "GG", "LG"):
            if d >= BLOCK_S and out and not out.endswith("… "):
                out = out.rstrip() + " … "
            continue
        if k == "H":
            if d >= PROLONG_S["H"]:
                tok = "Heeey"
            else:
                tok = "H" if d < 0.12 else "Hey"
            out += tok + ("-" if i < last_h else "")
        elif k == "G":
            if not seen_g and out and not out.endswith(" "):
                out += " "
            seen_g = True
            if i < last_g:
                tok = "G" if d < 0.10 else ("Gooo" if d >= PROLONG_S["G"] else "Goo")
                out += tok + "-"
            else:
                out += "Goooo" if d >= PROLONG_S["G"] else "Goo"
        elif k == "L":
            out += "gle"
    return " ".join(out.split()).replace("- ", "-")


def load_frame_model(path, kind: str):
    if kind in ("cnn", "crnn"):
        from .torch_models import TorchFrameModel
        return TorchFrameModel.load(path)
    if kind == "ensemble":
        # The ONNX export (scripts/export_onnx.py) gives the same posteriors without PyTorch; set WW_BACKEND=torch to
        # use the PyTorch checkpoints instead (for example right after retraining, before exporting again).
        import os
        from . import onnx_models
        if os.environ.get("WW_BACKEND", "").strip() != "torch" and onnx_models.available(Path(path).parent):
            return onnx_models.OnnxEnsembleModel.load(Path(path).parent)
        from .embtrain import EnsembleFrameModel
        return EnsembleFrameModel.load(path)
    if kind == "emb":
        from .embtrain import EmbFrameModel
        return EmbFrameModel.load(path)
    return FrameModel.load(path)


SOFT_MARGIN = 0.08      # hysteresis: the CONFIRMING check of a live wake-up may fall this far below the threshold


class WakeWordDetector:
    def __init__(self, model, verifier: Verifier | None = None):
        self.model = model
        self.verifier = verifier
        self.meta = {}

    @classmethod
    def load(cls, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)
        meta_p = model_dir / "meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {"frame_file": "frame_model.npz", "frame_type": "mlp"}
        model = load_frame_model(model_dir / meta["frame_file"], meta.get("frame_type", "mlp"))
        vp = model_dir / "verifier.json"
        ver = Verifier.from_json(json.loads(vp.read_text(encoding="utf-8"))) if vp.exists() else None
        det = cls(model, ver)
        det.meta = meta
        return det

    @property
    def gate(self):
        from .config import UNIT_GATE
        g = getattr(self.verifier, "unit_gate", None) if self.verifier else None
        return UNIT_GATE if g is None else float(g)

    @property
    def threshold(self):
        return self.verifier.threshold if self.verifier else 0.5

    # ---- scoring --------------------------------------------------------------
    def _score(self, dec) -> float:
        f = dec["features"]
        if f is None:
            return 0.0
        if self.verifier is not None:
            return self.verifier.score(f)
        return float(np.exp(-2.5 * f[0]))               # bootstrap fall-back (before a verifier is fitted)

    def score_wave(self, x: np.ndarray) -> float:
        lp = self.model.predict_wave(x)
        return self._score(decoder.decode(lp, self.gate))

    # ---- full analysis ----------------------------------------------------------
    def analyze(self, x: np.ndarray, repair: bool = False, with_curves: bool = True, live: bool = False) -> dict:
        x = np.asarray(x, dtype=np.float32)
        dur = len(x) / SR
        base = {"detected": False, "score": 0.0, "threshold": self.threshold, "duration": round(dur, 3),
                "keyword_text": KEYWORD_TEXT, "fluent_output": None, "heard_text": "", "span": None,
                "units": [], "events": [], "disfluent": False, "tail_silence_s": None, "soft": False, "soft_margin": SOFT_MARGIN}
        if len(x) < 2 * WIN or float(np.abs(x).max()) < 1e-4:
            base["reason"] = "silence"
            return base
        lp = self.model.predict_wave(x)
        dec = decoder.decode(lp, self.gate)
        score = self._score(dec)
        detected = bool(score >= self.threshold and dec["features"] is not None)
        base.update(score=round(score, 4), detected=detected)
        if with_curves:
            step = 2
            p = np.exp(lp)[::step]
            base["curves"] = {"step_s": step * HOP / SR, "classes": CLASSES,
                              "p": [[round(float(v), 3) for v in row] for row in p]}
        if dec.get("gated"):
            base["reason"] = "did not clearly hear: " + ", ".join(dec["gated"])
            return base
        if dec["span"] is None:
            base["reason"] = "no Hey-Goo-gle sequence found"
            return base
        a, b = dec["span"]
        # speech right BEFORE the phrase: a wake word starts after a pause, a 'hey ... google ... gle' that is
        # assembled out of ongoing talk (television, a conversation) does not.  Only enforced for live listening.
        am = np.argmax(lp, axis=1)
        lead_other = float((am[max(0, a - LEAD_FRAMES):a] == OTHER).mean()) if a >= 8 else 0.0
        base["lead_other"] = round(lead_other, 3)
        if live and detected and lead_other > LEAD_OTHER_MAX:
            detected = False
            base.update(detected=False, reason="speech right before the phrase (ongoing talk, not a wake word)")
        base["soft"] = bool(dec["features"] is not None and score >= self.threshold - SOFT_MARGIN
                            and not (live and lead_other > LEAD_OTHER_MAX))
        base["span"] = {"start": round(a * HOP / SR, 3), "end": round((b * HOP + WIN - HOP) / SR, 3)}
        base["tail_silence_s"] = round(max(0.0, (len(lp) - b) * HOP / SR), 3)
        units, events = [], []
        for kind, s, e in dec["runs"]:
            if kind in ("H", "G", "L", "HG", "GG", "LG") and s >= a and e <= b + 1:
                units.append((kind, s, e, (e - s) / 100.0))
        base["units"] = [{"kind": k, "label": UNIT_TEXT.get(k, "pause"), "start": round(s * HOP / SR, 3),
                          "end": round(e * HOP / SR, 3)} for k, s, e, d in units]
        n_h = sum(1 for u in units if u[0] == "H")
        n_g = sum(1 for u in units if u[0] == "G")
        for i, (k, s, e, d) in enumerate(units):
            t0, t1 = round(s * HOP / SR, 3), round(e * HOP / SR, 3)
            if k in PROLONG_S and d >= PROLONG_S[k]:
                events.append({"type": "prolongation", "word": "Hey" if k == "H" else "Google", "start": t0, "end": t1,
                               "duration_ms": int(d * 1000)})
            if k in ("HG", "GG") and d >= BLOCK_S:
                events.append({"type": "block", "word": "Hey" if k == "HG" else "Google", "start": t0, "end": t1,
                               "duration_ms": int(d * 1000)})
        if n_h > 1:
            hs = [u for u in units if u[0] == "H"]
            events.append({"type": "repetition", "word": "Hey", "count": n_h - 1,
                           "start": round(hs[0][1] * HOP / SR, 3), "end": round(hs[-1][2] * HOP / SR, 3)})
        if n_g > 1:
            gs = [u for u in units if u[0] == "G"]
            events.append({"type": "repetition", "word": "Google", "count": n_g - 1,
                           "start": round(gs[0][1] * HOP / SR, 3), "end": round(gs[-1][2] * HOP / SR, 3)})
        events.sort(key=lambda d: d["start"])
        base["events"] = events
        base["disfluent"] = bool(events)
        base["heard_text"] = _heard_text(units)
        if detected:
            base["fluent_output"] = KEYWORD_TEXT
            if repair:
                try:
                    rep = self.repair(x, dec)
                    base["repaired_audio"] = rep
                except Exception as exc:                      # the repair is a bonus: never fail the detection
                    base["repair_error"] = str(exc)
        return base

    # ---- fluent rendition ------------------------------------------------------------
    def repair(self, x: np.ndarray, dec) -> np.ndarray:
        """Re-assemble a fluent 'Hey Google' from the speaker's own audio: keep the final complete
        Hey / Goo / gle runs, drop repetitions and blocks, shorten prolongations."""
        runs = dec["runs"]
        pick = {}
        for kind, s, e in runs:
            if kind in ("H", "G", "L"):
                pick[kind] = (s, e)                        # last run of every kind
        if not all(k in pick for k in "HGL"):
            raise ValueError("incomplete path")
        db = frame_db(x)
        quiet = np.argsort(db)[: max(10, len(db) // 8)]
        noise_ref = np.concatenate([x[i * HOP:(i + 1) * HOP] for i in sorted(quiet)]) if len(quiet) else np.zeros(800, np.float32)
        rng = np.random.default_rng(0)
        caps = {"H": 0.30, "G": 0.34, "L": 0.40}
        parts = []
        for k in "HGL":
            s, e = pick[k]
            seg = x[max(0, s * HOP - HOP): min(len(x), e * HOP + WIN)].astype(np.float32)
            cap = int(caps[k] * SR)
            if len(seg) > cap:                              # shorten a prolonged sound: keep onset + offset
                head, tail = int(cap * 0.4), cap - int(cap * 0.4)
                seg = dsp.crossfade_concat([seg[:head], seg[-tail:]], xf_ms=12)
            parts.append(dsp.fade(seg, 4, 6))
            if k != "L":
                parts.append(dsp.make_noise_floor(noise_ref, int(0.03 * SR), rng))
        y = dsp.crossfade_concat(parts, xf_ms=6)
        lead = dsp.make_noise_floor(noise_ref, int(0.12 * SR), rng)
        y = np.concatenate([lead, y, lead])
        peak = float(np.abs(y).max())
        return (y * (0.85 / peak) if peak > 1e-6 else y).astype(np.float32)

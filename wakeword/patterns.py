"""Stuttering pattern analysis of real recordings, and a pattern sampler learned from it.

Three independent views of every clip are measured and compared (none of them is trusted alone):
  1. ACOUSTIC events  - energy/voicing segmentation + spectral steadiness + segment similarity (no model, no text),
  2. DETECTOR events  - how the wake-word decoder explains the clip (units, repeated units, blocks, prolongations),
  3. SPEECH-TO-TEXT   - literal transducer tokens (with times) and Whisper text (which normalises disfluency away).
The acoustic detector is validated on clips where the disfluency is injected by a known operation (recall / precision).

PatternModel fits a small parametric model of the observed disfluency (event probabilities, repetition counts, block
and prolongation durations, which part of the phrase is affected) with shrinkage towards literature priors, and
samples new disfluency from it for training-data augmentation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .config import SR
from .features import logmel
from . import robust
from .robust import voiced_segments

BLOCK_MIN_S = 0.25          # silence inside the phrase counted as a block
PAUSE_MIN_S = 0.12
PROLONG_MIN_S = 0.30        # steady voiced stretch counted as a prolongation
REP_MAX_SEG_S = 0.32        # a repeated fragment is short
REP_SIM = 0.88              # cosine similarity of consecutive segment onsets
DIP_MS = 40                 # energy dips shorter than this do not split a voiced segment
PROLONG_FLUX_REL = 0.5      # steady = spectral flux below this fraction of the clip's median voiced flux


def _onset_vec(lm, a, b, n=8):
    v = lm[a:min(b, a + n)].mean(0)
    return v - v.mean()


def acoustic_events(x):
    """Return a dict describing the disfluency-like structure of one clip (all times in seconds)."""
    x = np.asarray(x, np.float32)
    segs = voiced_segments(x)
    out = {"n_seg": len(segs), "dur": len(x) / SR, "speech_s": 0.0, "segs": [(a / SR, b / SR) for a, b in segs],
           "blocks": [], "pauses": [], "reps": 0, "rep_runs": [], "prolong": [], "rate_seg_per_s": 0.0}
    if not segs:
        return out
    out["speech_s"] = sum(b - a for a, b in segs) / SR
    for (a0, b0), (a1, b1) in zip(segs[:-1], segs[1:]):
        g = (a1 - b0) / SR
        if g >= BLOCK_MIN_S:
            out["blocks"].append(round(g, 3))
        elif g >= PAUSE_MIN_S:
            out["pauses"].append(round(g, 3))
    lm = logmel(x)
    hop = 160
    # repetitions: consecutive SHORT segments whose onsets look alike
    run = 1
    for i in range(1, len(segs)):
        a0, b0 = segs[i - 1]; a1, b1 = segs[i]
        s0, s1 = _onset_vec(lm, a0 // hop, b0 // hop), _onset_vec(lm, a1 // hop, b1 // hop)
        sim = float(s0 @ s1 / (np.linalg.norm(s0) * np.linalg.norm(s1) + 1e-9))
        short = (b0 - a0) / SR <= REP_MAX_SEG_S or (b1 - a1) / SR <= REP_MAX_SEG_S
        if sim >= REP_SIM and short and (a1 - b0) / SR < 0.6:
            out["reps"] += 1; run += 1
        else:
            if run > 1:
                out["rep_runs"].append(run - 1)
            run = 1
    if run > 1:
        out["rep_runs"].append(run - 1)
    # prolongation: long run of frames whose spectrum hardly changes while the voice is on
    flux = np.abs(np.diff(lm, axis=0)).mean(1) if len(lm) > 2 else np.zeros(0)
    voiced = np.zeros(len(flux), bool)
    for a, b in segs:
        voiced[min(a // hop, len(flux)):min(b // hop, len(flux))] = True
    ref = float(np.median(flux[voiced])) if voiced.any() else 0.0
    for a, b in segs:
        fa, fb = min(a // hop, len(flux)), min(b // hop, len(flux))
        steady = flux[fa:fb] < PROLONG_FLUX_REL * ref
        best = cur = 0
        for st in steady:
            cur = cur + 1 if st else 0
            best = max(best, cur)
        if best >= int(PROLONG_MIN_S * 100):
            out["prolong"].append(round(best / 100, 3))
    span = out["segs"][-1][1] - out["segs"][0][0]
    out["rate_seg_per_s"] = len(segs) / max(span, 0.2)
    return out


# ----------------------------------------------------------------------------- validation of the acoustic detector
def validate_acoustic(clips, ops, rng, n_rep=2):
    """Inject known operations into clips and check that the acoustic detector notices them.
    Returns {op: {"hit": k, "n": n, "false_on_clean": f}} - 'hit' means the matching event count increased."""
    key = {"prolongation": lambda e: len(e["prolong"]), "syllable repetition": lambda e: e["reps"],
           "word repetition": lambda e: e["reps"], "silent block": lambda e: len(e["blocks"])}
    base = [acoustic_events(c) for c in clips]
    res = {}
    for name, f in ops.items():
        hit = n = 0
        for c, b in zip(clips, base):
            for _ in range(n_rep):
                y = f(c, rng)
                e = acoustic_events(y)
                n += 1
                hit += int(key[name](e) > key[name](b))
        res[name] = {"hit": hit, "n": n}
    return res


# ----------------------------------------------------------------------------- speech-to-text features
def stt_features(rec):
    """rec: one entry of transcripts.json.  Literal-transducer tokens + Whisper text."""
    toks = [t for t in rec.get("zip_tokens", [])]
    words = rec.get("zip_text", "").split()
    wt = rec.get("whisper_text", "").lower().replace(",", " ").replace(".", " ").replace("?", " ").split()
    adj = sum(1 for a, b in zip(toks[:-1], toks[1:]) if a == b)
    repeated_words = sum(1 for a, b in zip(words[:-1], words[1:]) if a == b)
    loop = max((wt.count(w) for w in set(wt)), default=0)
    return {"n_tokens": len(toks), "n_words": len(words), "adjacent_same_token": adj, "adjacent_same_word": repeated_words,
            "whisper_max_word_count": loop, "whisper_loop_suspect": loop >= 6, "whisper_has_target": ("google" in wt and ("hey" in wt or "hay" in wt))}


# ----------------------------------------------------------------------------- pattern model
@dataclass
class PatternModel:
    """Parametric distribution of the disfluency of the people in the recordings (+ literature prior)."""
    p_event: dict = field(default_factory=lambda: {"prolongation": 0.25, "syllable repetition": 0.25, "word repetition": 0.15, "silent block": 0.25})
    rep_counts: list = field(default_factory=lambda: [1, 1, 2, 2, 3])          # extra repetitions per repetition event
    block_s: list = field(default_factory=lambda: [0.3, 0.5, 0.8, 1.2])      # silent block durations
    prolong_factor: list = field(default_factory=lambda: [1.8, 2.5, 3.5])    # stretch factor of the prolonged part
    p_initial: float = 0.5                                                    # event placed at the first voiced segment
    p_any: float = 0.8                                                        # probability that a variant gets disfluency at all
    source: str = "default"

    def apply(self, x, rng):
        """Sample disfluency from the model and apply it to waveform x (ops in sequence)."""
        out = np.asarray(x, np.float32)
        if rng.random() > self.p_any:
            return out
        n_done = 0
        for kind, pe in self.p_event.items():
            if rng.random() >= pe:
                continue
            if kind == "prolongation":
                out = robust.prolong(out, rng, factor=float(rng.choice(self.prolong_factor)), p_initial=self.p_initial)
            elif kind == "syllable repetition":
                out = robust.repeat_syllable(out, rng, n=int(rng.choice(self.rep_counts)), p_initial=self.p_initial)
            elif kind == "word repetition":
                out = robust.repeat_word(out, rng)
            else:
                out = robust.block(out, rng, dur=float(rng.choice(self.block_s)), p_initial=self.p_initial)
            n_done += 1
        return out[: int(10 * SR)]

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, s):
        return cls(**json.loads(s))


def shrink(emp, prior, n, k=8.0):
    """Posterior mean of a probability with `n` observations, shrunk towards `prior` with strength k."""
    return float((emp * n + prior * k) / (n + k))


def clip_stats(x):
    """Classification-free summary of a clip: [span, n_seg, total gap, max gap, speech fraction]."""
    e = acoustic_events(x); s = e["segs"]
    if not s:
        return None
    gaps = [b[0] - a[1] for a, b in zip(s[:-1], s[1:])]
    span = s[-1][1] - s[0][0]
    return np.array([span, e["n_seg"], sum(gaps), max(gaps, default=0.0), e["speech_s"] / max(span, 0.1)])


def energy_distance(A, B):
    """Energy distance between two sets of stat vectors (each column standardised by the pooled spread)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    sd = np.concatenate([A, B]).std(0) + 1e-9
    A, B = A / sd, B / sd
    d = lambda X, Y: np.mean(np.linalg.norm(X[:, None] - Y[None], axis=2))
    return 2 * d(A, B) - d(A, A) - d(B, B)


# ----------------------------------------------------------------------------- phrase-level restarts / repeats
def _group(segs, gap_s=0.15):
    g = []
    for a, b in segs:
        if g and (a - g[-1][1]) / SR < gap_s:
            g[-1][1] = b
        else:
            g.append([a, b])
    return g


def _dtw(A, B):
    n, m = len(A), len(B)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9); Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    D = 1 - An @ Bn.T
    C = np.full((n + 1, m + 1), np.inf); C[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            C[i, j] = D[i - 1, j - 1] + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
    return C[n, m] / (n + m)


RESTART_DTW = 0.10     # natural clips flagged 6% vs 51% of injected restarts (see run_pattern_study)


def restarts(x, thr=None):
    """Find groups of voiced segments (separated by >= 150 ms of silence) that sound like an earlier group:
    the speaker restarted or repeated the phrase / word.  Returns [(i, j, cost, start_i, start_j)] for consecutive-ish pairs.
    The earlier group may be a fragment of the later one ('hey si-' ... 'hey siri'): the shorter group is matched
    against the same-length opening of the longer one."""
    thr = RESTART_DTW if thr is None else thr
    x = np.asarray(x, np.float32)
    g = _group(voiced_segments(x))
    lm = logmel(x)
    lm = lm - lm.mean(0)          # remove the long-term spectrum: cosine on raw log-mel is ~0 for any two speech sounds
    out = []
    for i in range(len(g) - 1):
        j = i + 1
        a0, a1 = g[i][0] // 160, g[i][1] // 160
        b0, b1 = g[j][0] // 160, g[j][1] // 160
        if a1 - a0 < 6 or b1 - b0 < 6:
            continue
        A, B = lm[a0:a1], lm[b0:b1]
        L = min(len(A), len(B))
        c = min(_dtw(A[:L], B[:L]), _dtw(A, B) if 0.5 < len(A) / len(B) < 2 else 9)
        if c < thr:
            out.append((i, j, float(c), g[i][0] / SR, g[j][0] / SR))
    return out

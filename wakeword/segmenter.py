"""Find the spoken 'Hey Google' inside a noisy WhatsApp-style recording and split it into
sub-word units (Hey | Goo | gle) so that realistic disfluencies can be built from the speaker's
own audio.

Everything here is signal based (energy valleys + duration priors) so that it also works for
a new speaker that records a few fluent wake words in the browser.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import HOP, SR, WIN
from .features import frame_db, logmel


@dataclass
class Utterance:
    """A fluent 'Hey Google' with landmarks (all in samples of `audio`)."""
    audio: np.ndarray
    start: int            # first voiced sample
    end: int              # last voiced sample
    vowel_onset: int      # end of the /h/ aspiration in 'Hey'
    b1: int               # Hey | Goo boundary (energy valley)
    b2: int               # Goo | gle boundary (energy valley)
    noise: np.ndarray     # non-speech audio of the same recording (for realistic pauses)
    quality: float = 1.0
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- voiced regions
def voiced_regions(x: np.ndarray, min_frames: int = 6, merge_gap: float = 0.12):
    """Return [(start_frame, end_frame)] of speech-like activity (10 ms frames)."""
    db = frame_db(x)
    if db.size < 10:
        return [], db
    floor = np.percentile(db, 10)
    peak = db.max()
    thr = max(floor + 10.0, peak - 35.0)
    v = db > thr
    regs, s = [], None
    for i, b in enumerate(v):
        if b and s is None:
            s = i
        elif not b and s is not None:
            regs.append([s, i])
            s = None
    if s is not None:
        regs.append([s, len(v)])
    mg = int(merge_gap * 100)
    merged = []
    for a, b in regs:
        if merged and a - merged[-1][1] < mg:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    merged = [(a, b) for a, b in merged if b - a >= min_frames]
    return merged, db


def noise_reference(x: np.ndarray, regions, db: np.ndarray, pad: int = 8) -> np.ndarray:
    """Concatenate the quiet parts of a recording (used to synthesise believable silences)."""
    mask = np.ones(len(db), dtype=bool)
    for a, b in regions:
        mask[max(0, a - pad): b + pad] = False
    idx = np.where(mask)[0]
    if idx.size < 8:
        # no clean silence: use the quietest 15 % of frames
        order = np.argsort(db)[: max(8, int(0.15 * len(db)))]
        idx = np.sort(order)
    chunks = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            chunks.append((start, prev))
            start = i
        prev = i
    chunks.append((start, prev))
    parts = [x[a * HOP: b * HOP + WIN] for a, b in chunks if b - a >= 3]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


# ----------------------------------------------------------------------------- unit boundaries
def _smooth(v: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1:
        return v
    ker = np.ones(k) / k
    return np.convolve(np.pad(v, (k // 2, k // 2), mode="edge"), ker, mode="valid")[: len(v)]


def find_boundaries(x: np.ndarray, s: int, e: int):
    """Locate vowel onset, b1 (Hey|Goo) and b2 (Goo|gle) in samples. s,e are sample indices."""
    seg = x[s:e]
    db = frame_db(seg)
    n = len(db)
    if n < 25:
        third = max(1, (e - s) // 3)
        return s + 2 * HOP, s + third, s + 2 * third
    es = _smooth(db, 3)
    lo, hi = es.min(), es.max()
    en = (es - lo) / max(hi - lo, 1e-6)

    prior = [(0.27, 0.09), (0.38, 0.11), (0.35, 0.11)]
    min_f = 6
    ii = np.arange(n)[:, None].astype(np.float64)
    jj = np.arange(n)[None, :].astype(np.float64)
    f1, f2, f3 = ii / n, (jj - ii) / n, (n - jj) / n
    sc = (-((f1 - prior[0][0]) ** 2) / (2 * prior[0][1] ** 2)
          - ((f2 - prior[1][0]) ** 2) / (2 * prior[1][1] ** 2)
          - ((f3 - prior[2][0]) ** 2) / (2 * prior[2][1] ** 2)
          - 5.0 * (en[:, None] + en[None, :]))
    valid = (ii >= min_f) & (jj >= ii + min_f) & (jj <= n - min_f)
    sc = np.where(valid, sc, -1e18)
    flat = int(np.argmax(sc))
    i, j = divmod(flat, n)
    if sc[i, j] <= -1e17:
        i, j = int(0.27 * n), int(0.65 * n)

    # refine each boundary to the local energy minimum within +-3 frames
    def refine(k, lim_lo, lim_hi):
        a, b = max(lim_lo, k - 3), min(lim_hi, k + 4)
        return a + int(np.argmin(es[a:b])) if b > a else k
    i = refine(i, 3, n - 3)
    j = refine(j, i + 4, n - 3)

    # vowel onset in 'Hey': first frame within Hey reaching (max - 9 dB)
    hey_e = es[:i]
    thr = hey_e.max() - 9.0
    vo = int(np.argmax(hey_e >= thr)) if hey_e.size else 0
    vo = int(np.clip(vo, 1, max(1, i - 4)))
    return s + vo * HOP, s + i * HOP + WIN // 2, s + j * HOP + WIN // 2


def build_utterance(x: np.ndarray, region_samples=None, quality: float = 1.0, meta=None) -> Utterance | None:
    """Segment a recording that contains (at least) one 'Hey Google'.

    region_samples=(start,end) forces the utterance extent (used by the picker).
    """
    x = x.astype(np.float32)
    regs, db = voiced_regions(x)
    if not regs:
        return None
    noise = noise_reference(x, regs, db)
    if region_samples is None:
        a = regs[0][0]
        b = regs[-1][1]
        s, e = a * HOP, min(len(x), b * HOP + WIN)
    else:
        s, e = region_samples
    if e - s < int(0.3 * SR):
        return None
    vo, b1, b2 = find_boundaries(x, s, e)
    return Utterance(audio=x, start=int(s), end=int(e), vowel_onset=int(vo), b1=int(b1), b2=int(b2),
                     noise=noise, quality=quality, meta=meta or {})


# ----------------------------------------------------------------------------- picking the right window
def _dtw_distance(a: np.ndarray, b: np.ndarray, band: float = 0.35) -> float:
    """Length-normalised DTW distance between two feature sequences (cosine distance)."""
    na = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    nb = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    cost = 1.0 - na @ nb.T
    n, m = cost.shape
    inf = 1e9
    acc = np.full((n + 1, m + 1), inf)
    acc[0, 0] = 0.0
    w = int(max(abs(n - m), band * max(n, m)))
    for i in range(1, n + 1):
        j0, j1 = max(1, i - w), min(m, i + w)
        prev = acc[i - 1]
        row = acc[i]
        for j in range(j0, j1 + 1):
            row[j] = cost[i - 1, j - 1] + min(prev[j], prev[j - 1], row[j - 1])
    return float(acc[n, m] / (n + m))


def _dtw_feat(seg: np.ndarray) -> np.ndarray:
    lm = logmel(seg)
    lm = lm[::2]                                      # 20 ms resolution is enough and 4x faster
    lm = lm - lm.mean(axis=0, keepdims=True)         # cepstral-mean style normalisation
    return lm


class TemplateMatcher:
    """Scores how much an audio window sounds like a fluent 'Hey Google' (DTW to templates)."""

    def __init__(self, templates):
        self.feats = [_dtw_feat(t) for t in templates if len(t) > 0.4 * SR]

    def distance(self, seg: np.ndarray, exclude: int | None = None) -> float:
        f = _dtw_feat(seg)
        ds = sorted(_dtw_distance(f, t) for k, t in enumerate(self.feats) if k != exclude)
        return float(np.mean(ds[:5]))          # mean of the 5 closest templates

    def calibrate(self) -> float:
        """Leave-one-out distances of the templates -> acceptance threshold."""
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(self.feats))[: min(24, len(self.feats))]
        ds = []
        for k in idx:
            f = self.feats[k]
            d = sorted(_dtw_distance(f, t) for q, t in enumerate(self.feats) if q != k)
            ds.append(np.mean(d[:5]))
        return float(np.percentile(ds, 90) * 1.15)


def pick_wake_word_window(x: np.ndarray, matcher: TemplateMatcher | None = None,
                          max_len: float = 2.6, min_len: float = 0.5, pad_ms: int = 40):
    """Choose the best 'Hey Google' window inside a long / messy recording.

    Returns (start_sample, end_sample, score) or None. Without a matcher the longest plausible
    window is used.
    """
    regs, db = voiced_regions(x)
    if not regs:
        return None
    cands = set()
    for gap in (0.12, 0.25, 0.4, 0.6, 0.9):
        g = int(gap * 100)
        merged = []
        for a, b in regs:
            if merged and a - merged[-1][1] < g:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        for a, b in merged:
            d = (b - a) / 100.0
            if min_len <= d <= max_len:
                cands.add((a, b))
    # sliding combinations of consecutive raw regions (handles repeated attempts)
    for i in range(len(regs)):
        for j in range(i, min(len(regs), i + 4)):
            a, b = regs[i][0], regs[j][1]
            d = (b - a) / 100.0
            if min_len <= d <= max_len:
                cands.add((a, b))
    if not cands:
        return None
    best = None
    pad = int(pad_ms / 1000 * SR)
    for a, b in sorted(cands):
        s, e = max(0, a * HOP - pad), min(len(x), b * HOP + WIN + pad)
        if matcher is None:
            score = -(b - a)
        else:
            score = matcher.distance(x[s:e])
        if best is None or score < best[2]:
            best = (s, e, score)
    return best

"""Model-based forced alignment of fluent recordings.

The first-pass unit boundaries come from energy valleys + duration priors, which is noisy
(especially for clean TTS voices without a clear /g/ closure). Once a frame model exists, the
strict grammar  bg* HEY+ GOO+ GLE+ bg*  is aligned to every fluent source with Viterbi and the
Hey|Goo and Goo|gle boundaries are replaced by the acoustically consistent ones. Training on
the cleaner labels is the second pass.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import decoder
from .config import HOP, SR, WIN
from .features import frame_db
from .model import FrameModel
from .segmenter import Utterance


def refine_utterance(model: FrameModel, u: Utterance, margin_s: float = 0.15):
    """Return a copy of u with model-aligned b1 / b2 (or None if the alignment is implausible)."""
    a0 = max(0, u.start - int(margin_s * SR))
    b0 = min(len(u.audio), u.end + int(margin_s * SR))
    seg = u.audio[a0:b0]
    lp = model.predict_wave(seg)
    if lp.shape[0] < 20:
        return None
    score, path = decoder.viterbi(lp, decoder._STRICT)
    if path is None:
        return None
    kinds = np.array([decoder._STRICT.names[s][0] for s in path])
    g_idx, l_idx, h_idx = (np.where(kinds == k)[0] for k in ("G", "L", "H"))
    if not (g_idx.size and l_idx.size and h_idx.size):
        return None
    b1 = a0 + int(g_idx.min()) * HOP + WIN // 2
    b2 = a0 + int(l_idx.min()) * HOP + WIN // 2
    start = min(u.start, a0 + int(h_idx.min()) * HOP)
    end = u.end
    if not (start + int(0.05 * SR) < b1 < b2 < end - int(0.04 * SR)):
        return None
    # vowel onset of 'Hey' inside the new [start, b1] span (same energy rule as the first pass)
    db = frame_db(u.audio[start:b1])
    if db.size < 4:
        return None
    vo = start + int(np.clip(np.argmax(db >= db.max() - 9.0), 1, max(1, db.size - 3))) * HOP
    return replace(u, start=int(start), end=int(end), vowel_onset=int(vo), b1=int(b1), b2=int(b2),
                   meta={**u.meta, "aligned": True})


def align_keyword_clip(model, x: np.ndarray, max_outside_voiced: float = 0.15):
    """Frame labels for a clip KNOWN to contain one (possibly stuttered) 'Hey Google' - forced alignment.

    The disfluency grammar (repetitions, prolongations, blocks, filled pauses, restarts) is aligned to the clip with
    Viterbi; H / G / L frames become Hey / Goo / gle, pause states and the frames before / after become background.
    Because the transcript is known, this is forced alignment, not free recognition.  Returns (labels, span) or
    None when the alignment is implausible (too much speech outside the aligned phrase)."""
    from .config import BG, GLE, GOO, HEY
    from .negatives import vad_labels
    lp = model.predict_wave(x)
    if lp.shape[0] < 20:
        return None
    _, path = decoder.viterbi(lp, decoder._GRAMMAR)
    if path is None:
        return None
    kinds = np.array([decoder._GRAMMAR.names[s][0] for s in path])
    lab = np.full(len(kinds), BG, dtype=np.int64)
    lab[kinds == "H"], lab[kinds == "G"], lab[kinds == "L"] = HEY, GOO, GLE
    idx = np.where(np.isin(kinds, ["H", "G", "L"]))[0]
    if idx.size == 0:
        return None
    a, b = int(idx.min()), int(idx.max()) + 1
    for k, mn in (("H", 4), ("G", 4), ("L", 3)):
        if (kinds == k).sum() < mn:
            return None
    voiced = vad_labels(x)[: len(lab)] != BG
    if voiced.size < len(lab):
        voiced = np.pad(voiced, (0, len(lab) - voiced.size))
    outside = voiced.copy()
    outside[a:b] = False
    if voiced.sum() and outside.sum() / voiced.sum() > max_outside_voiced:
        return None
    # v1.2c: the labels must sit on the speech - every unit mostly voiced, little speech labelled as a pause
    for c in (HEY, GOO, GLE):
        if voiced[lab == c].mean() < 0.4:
            return None
    if voiced.sum() and (voiced & (lab == BG)).sum() / voiced.sum() > 0.2:
        return None
    return lab, (a, b)

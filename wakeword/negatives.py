"""Negative samples: everything that must NOT trigger the wake word.

Hard negatives are cut from the same speakers as the positives so that the model has to learn the
*phrase* (Hey -> Goo -> gle, in order) and not the speaker or the channel.
"""
from __future__ import annotations

import numpy as np

from . import dsp
from .config import BG, GLE, GOO, HEY, OTHER, HOP, SR, WIN
from .disfluency import DisfluencySynthesizer, Piece, Synth, _flatten
from .features import frame_db, n_frames
from .segmenter import Utterance

HARD_KINDS = ["hey_only", "hey_repeated", "hey_prolonged", "google_only", "google_stutter_only",
              "goo_incomplete", "google_then_hey", "google_google", "hey_goo_no_gle", "scrambled_units",
              "hey_other_word", "other_word_google", "hey_trailing_block"]
NOISE_KINDS = ["room_tone", "pink", "white", "hum", "clicks", "breaths", "beeps", "cough_burst", "keyboard"]


# ----------------------------------------------------------------------------- VAD-labelled external audio
def vad_labels(x: np.ndarray) -> np.ndarray:
    """OTHER for sound-activity frames, BG elsewhere (used for external negative audio)."""
    db = frame_db(x)
    if db.size == 0:
        return np.zeros(0, dtype=np.int64)
    floor, peak = np.percentile(db, 10), db.max()
    thr = max(floor + 8.0, peak - 32.0)
    act = (db > thr).astype(np.float64)
    act = np.convolve(act, np.ones(5) / 5, mode="same") > 0.4        # smooth flicker
    return np.where(act, OTHER, BG).astype(np.int64)


def labels_to_segments(lab: np.ndarray, tag="ext"):
    segs, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            segs.append((round((s * HOP) / SR, 4), round((i * HOP + (WIN - HOP)) / SR, 4), int(lab[s]), tag))
            s = i
    return segs


def external_pieces(x: np.ndarray, tag="other"):
    """Split an external clip into OTHER / BG pieces using the VAD."""
    lab = vad_labels(x)
    if lab.size == 0:
        return [Piece(x.astype(np.float32), OTHER, tag)]
    pieces, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            a, b = s * HOP, (len(x) if i == len(lab) else i * HOP)
            pieces.append(Piece(x[a:b].astype(np.float32), int(lab[s]), tag))
            s = i
    return pieces


def _finish(pieces, label, kind, spec=None) -> Synth:
    audio, placed = _flatten(pieces)
    segs = [(a / SR, b / SR, p.cls, p.tag) for p, (a, b) in placed]
    return Synth(audio=audio, segments=segs, events=[], label=label, disfluency_type=kind,
                 affected_word="none", params=spec or {})


# ----------------------------------------------------------------------------- hard negatives
def hard_negative(u: Utterance, kind: str, rng: np.random.Generator, other_clips=None) -> Synth:
    syn = DisfluencySynthesizer(u, rng)
    hey, _ = syn.unit("hey")
    goo, _ = syn.unit("goo")
    gle, _ = syn.unit("gle")
    P = lambda a, c, t="unit": Piece(a, c, t)
    gap = lambda lo, hi: syn.gap(float(rng.uniform(lo, hi)), "gap")
    lead = syn.gap(float(rng.uniform(120, 600)), "lead")
    trail = syn.gap(float(rng.uniform(250, 900)), "trail")
    body = []
    other = None
    if other_clips:
        other = other_clips[int(rng.integers(0, len(other_clips)))]
        other = other[: int(1.6 * SR)]

    if kind == "hey_only":
        body = [P(hey, HEY)]
    elif kind == "hey_repeated":
        n = int(rng.integers(2, 5))
        for k in range(n):
            seg = syn._vary(syn.unit("hey")[0], k, n, pitch_sd=0.8)
            body += [Piece(seg, HEY, "attempt")] + gap(60, 320)
    elif kind == "hey_prolonged":
        pre, st, post = syn.prolong("Hey", "vowel", float(rng.uniform(300, 1700)), -1)
        body = [pre, st, post]
    elif kind == "google_only":
        body = [P(goo, GOO), P(gle, GLE)]
    elif kind == "google_stutter_only":
        n = int(rng.integers(1, 4))
        body = syn.repetition("Google", str(rng.choice(["sound", "syllable"])), n, -1) + [P(goo, GOO), P(gle, GLE)]
    elif kind == "goo_incomplete":                       # "g-g-goo..." that never reaches 'gle'
        n = int(rng.integers(2, 5))
        body = syn.repetition("Google", str(rng.choice(["sound", "syllable"])), n, -1) + [P(goo, GOO)]
    elif kind == "google_then_hey":
        body = [P(goo, GOO), P(gle, GLE)] + gap(120, 500) + [P(hey, HEY)]
    elif kind == "google_google":
        g2 = syn._vary(np.concatenate([goo, gle]), 1, 2)
        body = [P(goo, GOO), P(gle, GLE)] + gap(100, 450) + [Piece(g2[: len(goo)], GOO, "attempt"), Piece(g2[len(goo):], GLE, "attempt")]
    elif kind == "hey_goo_no_gle":
        body = [P(hey, HEY)] + gap(0, 120) + [P(goo, GOO)]
    elif kind == "scrambled_units":
        order = list(rng.permutation(3))
        while order == [0, 1, 2]:
            order = list(rng.permutation(3))
        units = [(hey, HEY), (goo, GOO), (gle, GLE)]
        for o in order:
            body += [P(*units[o])] + gap(0, 120)
    elif kind == "hey_other_word" and other is not None:
        body = [P(hey, HEY)] + gap(60, 400) + external_pieces(other)
    elif kind == "other_word_google" and other is not None:
        body = external_pieces(other) + gap(60, 400) + [P(goo, GOO), P(gle, GLE)]
    elif kind == "hey_trailing_block":                   # blocked on 'Hey' and gave up
        body = syn.repetition("Hey", "sound", int(rng.integers(1, 4)), -1) + [P(hey, HEY)] + syn.gap(float(rng.uniform(1200, 2200)), "block")
    else:                                                # fallbacks when no external clip is available
        body = [P(hey, HEY)]
        kind = "hey_only"
    return _finish(lead + body + trail, "negative", kind)


# ----------------------------------------------------------------------------- non-speech negatives
def noise_negative(kind: str, rng: np.random.Generator, noise_ref=None):
    """Returns (audio, frame labels)."""
    n = int(SR * rng.uniform(1.5, 4.0))
    if kind == "room_tone":
        if noise_ref is not None and len(noise_ref) > 800 and dsp.rms(noise_ref) > 1e-5:
            x = dsp.make_noise_floor(noise_ref, n, rng)
        else:
            x = dsp.colored_noise(n, rng, 1.5) * 0.01
    elif kind in ("pink", "white"):
        x = dsp.colored_noise(n, rng, 1.0 if kind == "pink" else 0.0) * float(rng.uniform(0.01, 0.2))
    elif kind == "hum":
        t = np.arange(n) / SR
        f0 = float(rng.choice([50.0, 60.0, 100.0, 120.0]))
        x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in (1, 2, 3, 4)).astype(np.float32) * float(rng.uniform(0.02, 0.2))
    elif kind == "clicks":
        x = np.zeros(n, dtype=np.float32)
        for _ in range(int(rng.integers(2, 9))):
            p = int(rng.integers(0, n - 200))
            x[p:p + 120] += (rng.standard_normal(120) * np.exp(-np.arange(120) / 25.0)).astype(np.float32) * float(rng.uniform(0.2, 0.8))
        x += dsp.colored_noise(n, rng, 1.0) * 0.005
    elif kind == "breaths":
        x = np.zeros(n, dtype=np.float32)
        for _ in range(int(rng.integers(1, 3))):
            ln = int(SR * rng.uniform(0.3, 0.7))
            p = int(rng.integers(0, max(1, n - ln)))
            x[p:p + ln] += dsp.breath(ln, rng, level_db=float(rng.uniform(-28, -18)))[: n - p]
        x += dsp.colored_noise(n, rng, 1.0) * 0.004
    elif kind == "beeps":
        t = np.arange(n) / SR
        x = np.zeros(n, dtype=np.float32)
        for _ in range(int(rng.integers(1, 4))):
            f = float(rng.uniform(300, 3000))
            ln = int(SR * rng.uniform(0.1, 0.5))
            p = int(rng.integers(0, max(1, n - ln)))
            seg = np.sin(2 * np.pi * f * t[:ln]) * np.hanning(ln) * float(rng.uniform(0.05, 0.4))
            x[p:p + ln] += seg[: n - p].astype(np.float32)
        x += dsp.colored_noise(n, rng, 1.0) * 0.003
    elif kind == "cough_burst":
        x = np.zeros(n, dtype=np.float32)
        for _ in range(int(rng.integers(1, 4))):
            ln = int(SR * rng.uniform(0.08, 0.3))
            p = int(rng.integers(0, max(1, n - ln)))
            b = dsp.bandlimit(rng.standard_normal(ln).astype(np.float32), 200, 4000) * np.exp(-np.arange(ln) / (0.35 * ln))
            x[p:p + ln] += (b / (np.abs(b).max() + 1e-9) * float(rng.uniform(0.1, 0.6)))[: n - p]
        x += dsp.colored_noise(n, rng, 1.0) * 0.004
    else:   # keyboard
        x = np.zeros(n, dtype=np.float32)
        for _ in range(int(rng.integers(3, 14))):
            p = int(rng.integers(0, n - 400))
            ln = int(rng.integers(120, 320))
            b = dsp.bandlimit(rng.standard_normal(ln).astype(np.float32), 800, 7000) * np.exp(-np.arange(ln) / (0.2 * ln))
            x[p:p + ln] += b / (np.abs(b).max() + 1e-9) * float(rng.uniform(0.1, 0.5))
        x += dsp.colored_noise(n, rng, 1.0) * 0.004
    x = x.astype(np.float32)
    return x, vad_labels(x)

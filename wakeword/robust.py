"""Modulations of a voice (speaking rate, pitch, loudness, whisper, channel, truncation, ...) used both to TEST a model
('robustness battery') and to create extra real-voice training data from the recordings of real people.

Every transform maps a mono float32 waveform at 16 kHz to a waveform; none of them needs unit boundaries."""
from __future__ import annotations

import numpy as np

from . import dsp
from .config import SR


def speed(x, f, rng=None):
    """Speaking-rate change by factor f (>1 = faster), pitch preserved."""
    return x if abs(f - 1) < 0.02 else dsp.wsola(x.astype(np.float32), 1.0 / f, rng=rng or np.random.default_rng(0))


def pitch(x, semitones):
    return x if abs(semitones) < 0.1 else dsp.pitch_shift(x.astype(np.float32), semitones)[: len(x) + SR]


def whisper(x, mix=1.0, n_fft=512, hop=128, rng=None):
    """Whispered / breathy rendering: keep the short-time magnitude spectrum, randomise the phase (no voicing)."""
    rng = rng or np.random.default_rng(0)
    w = np.hanning(n_fft).astype(np.float32)
    pad = np.pad(x.astype(np.float32), (n_fft, n_fft))
    n = 1 + (len(pad) - n_fft) // hop
    idx = np.arange(n_fft)[None] + hop * np.arange(n)[:, None]
    X = np.fft.rfft(pad[idx] * w, axis=1)
    Y = np.abs(X) * np.exp(1j * rng.uniform(0, 2 * np.pi, X.shape))
    fr = np.fft.irfft(Y, n=n_fft, axis=1) * w
    out = np.zeros(len(pad), np.float32); norm = np.zeros(len(pad), np.float32)
    for i in range(n):
        out[i * hop:i * hop + n_fft] += fr[i]; norm[i * hop:i * hop + n_fft] += w * w
    out = (out / np.maximum(norm, 1e-6))[n_fft:-n_fft][: len(x)]
    out = out * (np.sqrt((x ** 2).mean()) / (np.sqrt((out ** 2).mean()) + 1e-9))
    return (mix * out + (1 - mix) * x[: len(out)]).astype(np.float32)


def level(x, peak_db):
    p = float(np.abs(x).max()) + 1e-9
    return (x * (10 ** (peak_db / 20) / p)).astype(np.float32)


def loud_clipped(x, drive=4.0):
    return dsp.soft_clip(level(x, 0.0) * drive, 2.0).astype(np.float32)


def noise(x, snr_db, kind="pink", rng=None):
    rng = rng or np.random.default_rng(0)
    n = dsp.colored_noise(len(x), rng, {"pink": 1.0, "white": 0.0, "brown": 2.0}[kind])
    return dsp.add_noise(x.astype(np.float32), n, snr_db).astype(np.float32)


def phone_band(x):
    return dsp.bandlimit(x.astype(np.float32), 300.0, 3400.0)


def room(x, rt60=0.35, wet=0.3, rng=None):
    return dsp.reverb(x.astype(np.float32), rng or np.random.default_rng(0), rt60=rt60, wet=wet)


def cut_head(x, ms):
    return x[int(ms * SR / 1000):]


def cut_tail(x, ms):
    return x[: max(len(x) - int(ms * SR / 1000), 1)]


def tremolo(x, hz=5.0, depth=0.5):
    t = np.arange(len(x)) / SR
    return (x * (1 - depth + depth * 0.5 * (1 + np.sin(2 * np.pi * hz * t)))).astype(np.float32)


def intonation(x, semis=3.0, rng=None):
    """Rising / falling intonation: pitch glide across the phrase (two halves shifted in opposite directions)."""
    h = len(x) // 2
    a, b = pitch(x[:h], -semis / 2), pitch(x[h:], semis / 2)
    return np.concatenate([a[:h], b[: len(x) - h]]).astype(np.float32)


# the battery: name -> function(x, rng) ; used by run.py robustness
BATTERY = {
    "original": lambda x, r: x,
    "speed x0.7 (slow)": lambda x, r: speed(x, 0.7, r),
    "speed x1.3": lambda x, r: speed(x, 1.3, r),
    "speed x1.6 (fast)": lambda x, r: speed(x, 1.6, r),
    "speed x1.9 (very fast)": lambda x, r: speed(x, 1.9, r),
    "speed x2.2 (rushed)": lambda x, r: speed(x, 2.2, r),
    "pitch -4 st": lambda x, r: pitch(x, -4),
    "pitch +4 st": lambda x, r: pitch(x, +4),
    "pitch +7 st (child-like)": lambda x, r: pitch(x, +7),
    "whisper": lambda x, r: whisper(x, 1.0, rng=r),
    "breathy (50% whisper)": lambda x, r: whisper(x, 0.5, rng=r),
    "quiet (-40 dBFS peak)": lambda x, r: level(x, -40),
    "loud + clipped": lambda x, r: loud_clipped(x),
    "pink noise 10 dB SNR": lambda x, r: noise(x, 10, "pink", r),
    "white noise 10 dB SNR": lambda x, r: noise(x, 10, "white", r),
    "pink noise 5 dB SNR": lambda x, r: noise(x, 5, "pink", r),
    "telephone band 300-3400 Hz": lambda x, r: phone_band(x),
    "reverb (RT60 0.5 s)": lambda x, r: room(x, 0.5, 0.35, r),
    "first 80 ms cut off": lambda x, r: cut_head(x, 80),
    "last 80 ms cut off": lambda x, r: cut_tail(x, 80),
    "tremolo (shaky voice)": lambda x, r: tremolo(x),
    "rising intonation": lambda x, r: intonation(x, 4.0),
    "falling intonation": lambda x, r: intonation(x, -4.0),
}


FAST_BIAS = 0.0       # share of speed changes drawn from the FAST range 1.4-2.4x (people who rush the phrase)


def modulate(x, rng, strength: float = 1.0):
    """Random voice modulation for TRAINING: speaking rate (weighted to include fast speech), pitch, whisper, shaky
    voice, intonation, band limit, reverb and a small onset / offset loss.  Channel noise and gain are added by the
    caller (augment.augment_channel).  Returns a new waveform; labels of a whole-clip positive stay valid."""
    y = x.astype(np.float32)
    if rng.random() < 0.7 * strength:
        lo, hi = (1.4, 2.4) if rng.random() < FAST_BIAS else (0.65, 2.2)
        y = speed(y, float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), rng)
    if rng.random() < 0.4 * strength:
        y = pitch(y, float(rng.uniform(-5.0, 7.0)))
    if rng.random() < 0.12 * strength:
        y = whisper(y, float(rng.uniform(0.5, 1.0)), rng=rng)
    if rng.random() < 0.12 * strength:
        y = tremolo(y, float(rng.uniform(3, 8)), float(rng.uniform(0.2, 0.6)))
    if rng.random() < 0.2 * strength and len(y) > SR:
        y = intonation(y, float(rng.choice([-1, 1]) * rng.uniform(2.0, 5.0)))
    if rng.random() < 0.15 * strength:
        y = phone_band(y)
    if rng.random() < 0.15 * strength:
        y = room(y, float(rng.uniform(0.2, 0.55)), float(rng.uniform(0.15, 0.4)), rng)
    if rng.random() < 0.1 * strength:
        y = cut_head(y, float(rng.uniform(20, 60))) if rng.random() < 0.5 else cut_tail(y, float(rng.uniform(20, 60)))
    return y.astype(np.float32)


# ----------------------------------------------------------------------------- simulated disfluencies on a REAL voice
# These need no alignment: an energy-based pass finds voiced segments (syllable-ish nuclei) and the operations act on them.
def voiced_segments(x, dip_ms=40, min_ms=50, rise=0.25):
    """Voiced segments with a threshold set from the clip's OWN noise floor: noise = 10th percentile of the frame energy (dB),
    peak = 95th percentile, threshold = noise + rise*(peak - noise) but at least 6 dB above the noise."""
    hop, win = 160, 400
    x = np.asarray(x, np.float32)
    if len(x) < win:
        return []
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None] + hop * np.arange(n)[:, None]
    e = 10 * np.log10((x[idx] ** 2).mean(1) + 1e-10)
    noise, peak = np.percentile(e, 10), np.percentile(e, 95)
    if peak - noise < 8:
        return []
    on = e > noise + max(6.0, rise * (peak - noise))
    segs, s0 = [], None
    for i, v in enumerate(on):
        if v and s0 is None:
            s0 = i
        if not v and s0 is not None:
            segs.append([s0, i]); s0 = None
    if s0 is not None:
        segs.append([s0, len(on)])
    merged = []
    for a, b in segs:
        if merged and a - merged[-1][1] < dip_ms // 10:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a * hop, (b + 2) * hop) for a, b in merged if (b - a) * 10 >= min_ms]


def _voiced_segments(x, hop=160, min_ms=50, dip_ms=40, rel_db=-28.0):
    return voiced_segments(x, dip_ms=dip_ms, min_ms=min_ms)


def _pick(n, rng, p_initial):
    """Index of the segment an event is placed on: the first one with probability p_initial, else uniform."""
    if p_initial is not None and rng.random() < p_initial:
        return 0
    return int(rng.integers(0, n))


def _floor(x, n, rng):
    """Background noise of length n built ONLY from the clip's own non-speech frames (the quietest ~25 % of frames).
    (v1.5 fix: it used to draw from the whole clip, i.e. it inserted scrambled speech where silence was intended.)"""
    if n <= 0:
        return np.zeros(0, np.float32)
    x = np.asarray(x, np.float32)
    hop, win = 160, 400
    if len(x) < win * 3:
        return np.zeros(n, np.float32)
    k = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None] + hop * np.arange(k)[:, None]
    e = (x[idx] ** 2).mean(1)
    quiet = np.flatnonzero(e <= np.percentile(e, 25))
    ref = np.concatenate([x[i * hop:i * hop + win] for i in quiet]) if len(quiet) else np.zeros(0, np.float32)
    if len(ref) < int(0.2 * SR):
        return (rng.normal(0, np.sqrt(np.percentile(e, 10) + 1e-12), n)).astype(np.float32)
    return dsp.make_noise_floor(ref.astype(np.float32), n, rng)


def prolong(x, rng, factor=None, p_initial=None):
    """Stretch one stable part inside a voiced segment (a sound prolongation, e.g. 'Heeeey', 'Gooooogle')."""
    segs = _voiced_segments(x)
    if not segs:
        return x
    a, b = segs[_pick(len(segs), rng, p_initial)]
    L = b - a
    if L < 0.12 * SR:
        return x
    w = int(min(L * 0.5, rng.uniform(0.08, 0.2) * SR))
    s = a + int(rng.uniform(0.15, 0.55) * (L - w)) if L > w else a
    f = factor or float(np.exp(rng.uniform(np.log(1.8), np.log(4.5))))
    mid = dsp.wsola(x[s:s + w].astype(np.float32), f, rng=rng)
    return dsp.crossfade_concat([x[:s], mid, x[s + w:]], 8.0).astype(np.float32)


def repeat_syllable(x, rng, n=None, p_initial=None):
    """Repeat the start of a voiced segment: 'h-h-hey', 'goo-goo-google'."""
    segs = _voiced_segments(x)
    if not segs:
        return x
    a, b = segs[_pick(min(len(segs), 3), rng, p_initial)]
    L = int(rng.uniform(0.05, 0.16) * SR)
    L = min(L, (b - a) // 2) if b - a > 0.1 * SR else L
    if L < 0.03 * SR:
        return x
    n = n or int(rng.integers(1, 5))
    chunk = dsp.fade(x[a:a + L].astype(np.float32), 4, 6)
    parts = [x[:a]]
    for _ in range(n):
        parts += [chunk, _floor(x, int(rng.uniform(0.02, 0.14) * SR), rng)]
    parts.append(x[a:])
    return dsp.crossfade_concat(parts, 4.0).astype(np.float32)


def repeat_word(x, rng, n=None):
    """Repeat the first stretch of speech ('Hey, hey, hey Google')."""
    segs = _voiced_segments(x)
    if len(segs) == 0:
        return x
    a, b = segs[0]
    end = b if len(segs) < 2 or (b - a) > 0.25 * SR else segs[1][1]
    end = min(end, a + int(0.6 * SR))
    chunk = dsp.fade(x[a:end].astype(np.float32), 5, 8)
    parts = [x[:a]]
    for _ in range(n or int(rng.integers(1, 3))):
        parts += [chunk, _floor(x, int(rng.uniform(0.05, 0.35) * SR), rng)]
    parts.append(x[a:])
    return dsp.crossfade_concat(parts, 4.0).astype(np.float32)


def block(x, rng, dur=None, p_initial=None):
    """Silent block inside the phrase (a stuck articulation) at a point between or inside voiced segments."""
    segs = _voiced_segments(x)
    if len(segs) < 1:
        return x
    if len(segs) >= 2 and rng.random() < 0.6:
        k = _pick(len(segs) - 1, rng, p_initial); p = (segs[k][1] + segs[k + 1][0]) // 2
    else:
        a, b = segs[_pick(len(segs), rng, p_initial)]; p = a + int(rng.uniform(0.3, 0.7) * (b - a))
    d = int((dur or rng.uniform(0.25, 1.3)) * SR)
    return dsp.crossfade_concat([x[:p], _floor(x, d, rng), x[p:]], 6.0).astype(np.float32)


DISFLUENCIES = {"prolongation": prolong, "syllable repetition": repeat_syllable, "word repetition": repeat_word, "silent block": block}


def restart(x, rng, frac=None, pause=None, p_initial=None):
    """Phrase restart: the speaker gets part-way ('hey si-'), stops, and says the whole phrase again ('hey siri')."""
    segs = _voiced_segments(x)
    if not segs:
        return x
    end = segs[-1][1]
    cut_lo = segs[0][0] + int(0.12 * SR)
    cut = cut_lo + int((frac if frac is not None else rng.uniform(0.2, 0.8)) * max(end - cut_lo, 1))
    gap = int((pause if pause is not None else rng.uniform(0.15, 0.9)) * SR)
    head = x[segs[0][0]:cut].astype(np.float32)
    lead = x[:segs[0][0]]
    return dsp.crossfade_concat([lead, head, _floor(x, gap, rng), x[segs[0][0]:]], 6.0).astype(np.float32)[: int(10 * SR)]


DISFLUENCIES["phrase restart"] = restart


PATTERN_MODEL = None      # a patterns.PatternModel fitted to real recordings; used for half of the augmentations when set


def disfluent(x, rng, kinds=None, max_ops=2):
    """1-2 random disfluency operations applied in sequence (a 'mixed' stutter).
    With a fitted PATTERN_MODEL, half of the calls (when no kinds are forced) sample from it instead."""
    if PATTERN_MODEL is not None and kinds is None and rng.random() < 0.5:
        return PATTERN_MODEL.apply(x, rng)
    names = list(kinds or DISFLUENCIES)
    out = x.astype(np.float32)
    for _ in range(int(rng.integers(1, max_ops + 1))):
        out = DISFLUENCIES[names[int(rng.integers(0, len(names)))]](out, rng)
    return out[: int(10 * SR)]


BATTERY_DISFLUENCY = {k: (lambda x, rng, f=f: f(x, rng)) for k, f in DISFLUENCIES.items()}
BATTERY_DISFLUENCY["mixed stutter"] = lambda x, rng: disfluent(x, rng, max_ops=3)

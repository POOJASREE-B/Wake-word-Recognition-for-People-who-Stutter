"""Speaker perturbation and recording-channel augmentation."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import dsp
from .config import SR
from .features import frame_db
from .segmenter import Utterance


# v1.2e: speaking-rate range of real use. v1.1-v1.2d used 0.88-1.15x (0.82-1.22x for real voices), so a slow,
# deliberate or a quick 'Hey Google' was outside everything the model had seen. Overall tempo is log-uniform in
# TEMPO_RANGE (slow and fast equally likely); each unit (Hey / Goo / gle) gets its own extra +-UNIT_JITTER, because
# people often stretch or rush just one part ('Hey Gooogle').
TEMPO_RANGE = (0.65, 1.55)
UNIT_JITTER = 0.25


def perturb_utterance(u: Utterance, rng: np.random.Generator, tempo=None, semitones=None, wide: bool = False) -> Utterance:
    """Change speaking rate (overall and per unit) and pitch of the spoken region; unit landmarks are recomputed
    from the stretched pieces, so the Hey / Goo / gle labels stay exact."""
    if tempo is None:
        tempo = float(np.exp(rng.uniform(np.log(TEMPO_RANGE[0]), np.log(TEMPO_RANGE[1]))))
    semitones = float(rng.uniform(-3.5, 3.5) if wide else rng.uniform(-2.0, 2.0)) if semitones is None else semitones
    x, s, e = u.audio, u.start, u.end
    cuts = [s, u.b1, u.b2, e]
    if not (s < u.b1 < u.b2 < e):
        cuts = [s, e]
    pieces, lens = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        seg = x[a:b].astype(np.float32)
        t = tempo * (float(rng.uniform(1 - UNIT_JITTER, 1 + UNIT_JITTER)) if len(cuts) == 4 else 1.0)
        if abs(t - 1.0) > 0.02 and len(seg) > 400:
            seg = dsp.wsola(seg, 1.0 / t, rng=rng)
        pieces.append(seg)
        lens.append(len(seg))
    region2 = np.concatenate(pieces)
    if abs(semitones) > 0.1:
        region2 = dsp.pitch_shift(region2, semitones)[: len(region2)]
    audio = np.concatenate([x[:s], region2, x[e:]]).astype(np.float32)
    if len(cuts) == 4:
        b1, b2 = s + lens[0], s + lens[0] + lens[1]
    else:
        sc = len(region2) / max(e - s, 1)
        b1, b2 = int(round(s + (u.b1 - s) * sc)), int(round(s + (u.b2 - s) * sc))
    vo = s + int(round((u.vowel_onset - s) * lens[0] / max(cuts[1] - s, 1)))
    return replace(u, audio=audio, end=s + len(region2), vowel_onset=min(max(vo, s), b1), b1=b1, b2=b2,
                   meta={**u.meta, "tempo": round(tempo, 3), "semitones": round(semitones, 2)})


def babble(pool, rng: np.random.Generator, n: int, k: int = 3) -> np.ndarray:
    """Overlapping speech from other clips (cocktail-party style background)."""
    out = np.zeros(n, dtype=np.float32)
    if not pool:
        return out
    for _ in range(k):
        clip = pool[int(rng.integers(0, len(pool)))]
        if len(clip) < 1000:
            continue
        reps = int(np.ceil(n / len(clip))) + 1
        c = np.tile(clip, reps)
        off = int(rng.integers(0, max(1, len(c) - n)))
        seg = c[off: off + n]
        out += seg / (dsp.rms(seg) + 1e-6)
    return out / (np.abs(out).max() + 1e-9)


def augment_channel(x: np.ndarray, rng: np.random.Generator, babble_pool=None, noise_ref=None, strength: float = 1.0):
    """Random gain / band-limit / reverb / noise / clipping. Returns (y, info)."""
    info = {}
    y = dsp.highpass(x.astype(np.float32), 50.0)
    g = float(rng.uniform(-14, 6))
    y = y * dsp.db_to_gain(g)
    info["gain_db"] = round(g, 1)
    if rng.random() < 0.25 * strength:
        lo, hi = float(rng.uniform(80, 300)), float(rng.uniform(3400, 7200))
        y = dsp.bandlimit(y, lo, hi)
        info["bandlimit"] = [round(lo), round(hi)]
    if rng.random() < 0.2 * strength:
        y = dsp.reverb(y, rng, rt60=float(rng.uniform(0.12, 0.45)), wet=float(rng.uniform(0.15, 0.4)))
        info["reverb"] = True
    if rng.random() < 0.7 * strength:
        kind = str(rng.choice(["pink", "white", "brown", "hum", "babble", "room"], p=[0.25, 0.1, 0.15, 0.1, 0.25, 0.15]))
        n = len(y)
        if kind == "babble" and babble_pool:
            noise = babble(babble_pool, rng, n)
        elif kind == "room" and noise_ref is not None and len(noise_ref) > 800 and dsp.rms(noise_ref) > 1e-5:
            noise = dsp.make_noise_floor(noise_ref, n, rng)
        elif kind == "hum":
            t = np.arange(n) / SR
            f0 = float(rng.choice([50.0, 60.0]))
            noise = sum(np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28)) / h for h in (1, 2, 3, 5)).astype(np.float32)
            noise += 0.3 * dsp.colored_noise(n, rng, 1.0)
        else:
            noise = dsp.colored_noise(n, rng, {"pink": 1.0, "white": 0.0, "brown": 2.0}.get(kind, 1.0))
        snr = float(rng.uniform(5, 35))
        y = dsp.add_noise(y, noise, snr)
        info["noise"] = kind
        info["snr_db"] = round(snr, 1)
    if rng.random() < 0.08 * strength:
        y = dsp.soft_clip(y * float(rng.uniform(2.0, 5.0)), 2.0)
        info["clip"] = True
    peak = float(np.abs(y).max())
    if peak > 0.97:
        y = y * (0.97 / peak)
    return y.astype(np.float32), info

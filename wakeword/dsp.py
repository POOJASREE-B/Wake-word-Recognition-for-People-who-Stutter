"""Small DSP toolbox: WSOLA time-stretch, pitch shift, noise floors, crossfades, channel effects."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

from .config import SR


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12)) if len(x) else 0.0


def fade(x: np.ndarray, fade_in_ms: float = 0, fade_out_ms: float = 0, sr: int = SR) -> np.ndarray:
    y = x.astype(np.float32).copy()
    a, b = int(sr * fade_in_ms / 1000), int(sr * fade_out_ms / 1000)
    if a > 0 and len(y) > 0:
        a = min(a, len(y))
        y[:a] *= np.sin(np.linspace(0, np.pi / 2, a, dtype=np.float32)) ** 2
    if b > 0 and len(y) > 0:
        b = min(b, len(y))
        y[-b:] *= np.cos(np.linspace(0, np.pi / 2, b, dtype=np.float32)) ** 2
    return y


def snap_zero(x: np.ndarray, i: int, radius: int = 80) -> int:
    """Move cut point i to the nearest zero crossing (avoids clicks)."""
    i = int(np.clip(i, 1, max(1, len(x) - 2)))
    lo, hi = max(1, i - radius), min(len(x) - 1, i + radius)
    seg = x[lo - 1:hi]
    sign = np.signbit(seg)
    zc = np.where(sign[1:] != sign[:-1])[0]
    if zc.size == 0:
        return i
    return int(lo + zc[np.argmin(np.abs(lo + zc - i))])


def crossfade_concat(parts, xf_ms: float = 6.0, sr: int = SR) -> np.ndarray:
    """Equal-power crossfade concatenation of audio parts."""
    parts = [p.astype(np.float32) for p in parts if len(p) > 0]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    out = parts[0]
    for p in parts[1:]:
        n = int(min(sr * xf_ms / 1000, len(out), len(p)))
        if n < 2:
            out = np.concatenate([out, p])
            continue
        t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
        mixed = out[-n:] * np.cos(t) + p[:n] * np.sin(t)
        out = np.concatenate([out[:-n], mixed, p[n:]])
    return out


# ----------------------------------------------------------------------------- WSOLA
def wsola(x: np.ndarray, stretch: float, sr: int = SR, win_ms: float = 30.0, tol_ms: float = 9.0,
          rng: np.random.Generator | None = None, jitter_ms: float = 0.0, overlap: int = 2) -> np.ndarray:
    """Waveform-similarity overlap-add time-scale modification.

    stretch = output_length / input_length (>1 slows down / prolongs). Pitch is preserved.
    `jitter_ms` randomly perturbs the analysis position so a prolonged vowel does not become a
    perfectly periodic loop (that would sound like a digital 'buzz', not a held sound).
    """
    x = x.astype(np.float32)
    rng = rng or np.random.default_rng(0)
    N = int(sr * win_ms / 1000)
    N += N % 2
    Hs = N // overlap
    if len(x) < 2 * N or abs(stretch - 1.0) < 0.02:
        target = int(len(x) * stretch)
        if target <= 0:
            return x[:0]
        if len(x) == 0:
            return x
        idx = np.linspace(0, len(x) - 1, target)
        return np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    tol = int(sr * tol_ms / 1000)
    Ha = Hs / stretch
    n_out = int(np.ceil(len(x) * stretch / Hs)) + 1
    win = np.hanning(N + 1)[:-1].astype(np.float32)
    xp = np.pad(x, (tol + N, tol + 2 * N))
    base = tol + N
    out = np.zeros(n_out * Hs + N, dtype=np.float32)
    wsum = np.zeros_like(out)
    prev = None
    jit = int(sr * jitter_ms / 1000)
    for i in range(n_out):
        nominal = int(round(i * Ha))
        if jit:
            nominal += int(rng.integers(-jit, jit + 1))
        nominal = int(np.clip(nominal, 0, len(x) - 1))
        if prev is None:
            start = base + nominal
        else:
            target = xp[prev + Hs: prev + Hs + N]
            lo = max(tol, base + nominal - tol)
            hi = min(len(xp) - N - 1, base + nominal + tol)
            if hi <= lo:
                start = int(np.clip(base + nominal, 0, len(xp) - N - 1))
            else:
                seg = xp[lo: hi + N]
                corr = np.correlate(seg, target, mode="valid")
                energy = np.convolve(seg * seg, np.ones(N, dtype=np.float32), mode="valid")
                score = corr / np.sqrt(energy + 1e-6)
                start = lo + int(np.argmax(score))
        out[i * Hs: i * Hs + N] += win * xp[start: start + N]
        wsum[i * Hs: i * Hs + N] += win
        prev = start
    out = out / np.maximum(wsum, 1e-3)
    return out[: int(len(x) * stretch)].astype(np.float32)


def stretch_to(x: np.ndarray, target_len: int, **kw) -> np.ndarray:
    if len(x) == 0 or target_len <= 0:
        return x[:0]
    y = wsola(x, target_len / len(x), **kw)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="edge" if len(y) else "constant")
    return y[:target_len]


def pitch_shift(x: np.ndarray, semitones: float, sr: int = SR) -> np.ndarray:
    """Pitch shift keeping duration: resample (changes pitch+length), then WSOLA back."""
    if abs(semitones) < 0.05 or len(x) < 800:
        return x.astype(np.float32)
    f = 2.0 ** (semitones / 12.0)
    fr = Fraction(f).limit_denominator(64)
    y = resample_poly(x, fr.denominator, fr.numerator).astype(np.float32)
    return stretch_to(y, len(x), sr=sr)


# ----------------------------------------------------------------------------- noise / channel
def make_noise_floor(ref: np.ndarray, n: int, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Background noise of length n built from the recording's own non-speech audio.

    Random chunks of `ref` are cross-faded so no periodic loop is audible. If the source has
    exact digital silence we reproduce that (it is what the recording really sounds like).
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if ref is None or len(ref) < int(0.05 * sr) or rms(ref) < 1e-6:
        return np.zeros(n, dtype=np.float32)
    out = np.zeros(n + sr // 2, dtype=np.float32)
    pos = 0
    chunk = max(int(0.06 * sr), min(len(ref), int(rng.uniform(0.08, 0.2) * sr)))
    xf = min(chunk // 3, int(0.02 * sr))
    prev_tail = None
    while pos < n:
        s = int(rng.integers(0, max(1, len(ref) - chunk + 1)))
        seg = ref[s: s + chunk].astype(np.float32).copy()
        seg -= seg.mean()
        if prev_tail is None:
            out[pos: pos + len(seg)] += seg
            pos += len(seg) - xf
        else:
            w = np.sin(np.linspace(0, np.pi / 2, xf, dtype=np.float32))
            seg[:xf] *= w
            out[pos: pos + len(seg)] += seg
            pos += len(seg) - xf
        prev_tail = True
        w2 = np.cos(np.linspace(0, np.pi / 2, xf, dtype=np.float32))
        out[pos: pos + xf] *= w2
    return out[:n]


def breath(n: int, rng: np.random.Generator, sr: int = SR, level_db: float = -34.0) -> np.ndarray:
    """Audible inhalation / aspirated tension: band-shaped noise with a soft envelope."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    noise = rng.standard_normal(n).astype(np.float32)
    sos = butter(2, [300, 3800], btype="band", fs=sr, output="sos")
    y = sosfilt(sos, noise).astype(np.float32)
    env = np.clip(np.sin(np.linspace(0, np.pi, n, dtype=np.float32)), 0.0, None) ** 1.5
    y = y * env
    return y / (rms(y) + 1e-9) * db_to_gain(level_db)


def colored_noise(n: int, rng: np.random.Generator, beta: float = 1.0) -> np.ndarray:
    """1/f^beta noise (0 white, 1 pink, 2 brown)."""
    n = max(int(n), 2)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    spec = spec / f ** (beta / 2.0)
    y = np.fft.irfft(spec, n).astype(np.float32)
    return y / (np.abs(y).max() + 1e-9)


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(x):
        reps = int(np.ceil(len(x) / max(len(noise), 1)))
        noise = np.tile(noise, reps)
    noise = noise[: len(x)]
    ps, pn = rms(x), rms(noise)
    if pn < 1e-9 or ps < 1e-9:
        return x
    gain = ps / (pn * db_to_gain(snr_db))
    return (x + gain * noise).astype(np.float32)


def reverb(x: np.ndarray, rng: np.random.Generator, rt60: float = 0.25, sr: int = SR, wet: float = 0.35) -> np.ndarray:
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0
    ir = ir.astype(np.float32)
    y = fftconvolve(x, ir)[: len(x)].astype(np.float32)
    y = y / (rms(y) + 1e-9) * rms(x)
    return ((1 - wet) * x + wet * y).astype(np.float32)


def bandlimit(x: np.ndarray, lo: float, hi: float, sr: int = SR) -> np.ndarray:
    hi = min(hi, sr / 2 - 100)
    sos = butter(4, [max(lo, 20.0), hi], btype="band", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def highpass(x: np.ndarray, fc: float = 60.0, sr: int = SR) -> np.ndarray:
    sos = butter(2, fc, btype="high", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def soft_clip(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    return (np.tanh(x * drive) / np.tanh(drive)).astype(np.float32)

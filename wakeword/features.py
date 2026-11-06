"""Log-mel features implemented in plain numpy (no librosa / torch needed)."""
from __future__ import annotations

import numpy as np

from .config import CONTEXT, FMAX, FMIN, HOP, N_FFT, N_MELS, SR, WIN

_WINDOW = np.hanning(WIN + 1)[:-1].astype(np.float32)


def _hz2mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel2hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def _mel_filterbank() -> np.ndarray:
    mels = np.linspace(_hz2mel(FMIN), _hz2mel(FMAX), N_MELS + 2)
    hz = _mel2hz(mels)
    bins = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    fb = np.zeros((N_MELS, bins.size), dtype=np.float32)
    for i in range(N_MELS):
        lo, ce, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / max(ce - lo, 1e-6)
        down = (hi - bins) / max(hi - ce, 1e-6)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    fb /= np.maximum(fb.sum(axis=1, keepdims=True), 1e-6)
    return fb


_FB = _mel_filterbank()


def n_frames(n_samples: int) -> int:
    return 0 if n_samples < WIN else 1 + (n_samples - WIN) // HOP


def frame_signal(x: np.ndarray) -> np.ndarray:
    n = n_frames(len(x))
    if n == 0:
        return np.zeros((0, WIN), dtype=np.float32)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def frame_db(x: np.ndarray) -> np.ndarray:
    """Frame RMS in dBFS (one value / 10 ms)."""
    fr = frame_signal(x)
    if fr.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    return (10.0 * np.log10((fr.astype(np.float64) ** 2).mean(axis=1) + 1e-10)).astype(np.float32)


def soft_agc(x: np.ndarray, target_db: float = -20.0, floor_db: float = -52.0, max_gain_db: float = 30.0) -> np.ndarray:
    """Level normalisation used identically in training and at run time.

    The 98th percentile of frame energy is moved to `target_db`, but only if the clip
    contains real signal (> floor_db), so pure noise is never amplified into 'speech'.
    """
    db = frame_db(x)
    if db.size == 0:
        return x.astype(np.float32)
    p = float(np.percentile(db, 98))
    if p < floor_db:
        return x.astype(np.float32)
    gain_db = float(np.clip(target_db - p, -30.0, max_gain_db))
    return (x * (10.0 ** (gain_db / 20.0))).astype(np.float32)


def mel_power(x: np.ndarray) -> np.ndarray:
    """(T, N_MELS) mel power spectrogram of the int16-scaled signal."""
    fr = frame_signal((x * 32768.0).astype(np.float32))
    if fr.shape[0] == 0:
        return np.zeros((0, N_MELS), dtype=np.float64)
    fr = fr - fr.mean(axis=1, keepdims=True)
    fr = np.concatenate([fr[:, :1], fr[:, 1:] - 0.97 * fr[:, :-1]], axis=1) * _WINDOW
    spec = np.abs(np.fft.rfft(fr, n=N_FFT, axis=1)) ** 2
    return (spec @ _FB.T).astype(np.float64)


def pcen(x: np.ndarray, s: float = 0.04, alpha: float = 0.98, delta: float = 2.0, r: float = 0.5, eps: float = 1e-6) -> np.ndarray:
    """Per-channel energy normalisation: automatic gain control + stationary-noise / channel suppression.

    This is what makes the same model usable for WhatsApp recordings, browser microphones and TTS."""
    E = mel_power(x)
    if E.shape[0] == 0:
        return E.astype(np.float32)
    M = np.empty_like(E)
    M[0] = E[0]
    for t in range(1, len(E)):
        M[t] = (1.0 - s) * M[t - 1] + s * E[t]
    out = (E / (eps + M) ** alpha + delta) ** r - delta ** r
    return out.astype(np.float32)


def logmel(x: np.ndarray) -> np.ndarray:
    """(T, N_MELS) log-mel spectrogram, 25 ms window / 10 ms hop."""
    fr = frame_signal(x.astype(np.float32))
    if fr.shape[0] == 0:
        return np.zeros((0, N_MELS), dtype=np.float32)
    fr = fr - fr.mean(axis=1, keepdims=True)
    fr = np.concatenate([fr[:, :1], fr[:, 1:] - 0.97 * fr[:, :-1]], axis=1) * _WINDOW
    spec = np.abs(np.fft.rfft(fr, n=N_FFT, axis=1)) ** 2
    mel = spec @ _FB.T
    return np.log(mel + 1e-7).astype(np.float32)


def stack_context(feat: np.ndarray, offsets=CONTEXT) -> np.ndarray:
    """Concatenate neighbouring frames -> (T, len(offsets)*N_MELS)."""
    T = feat.shape[0]
    if T == 0:
        return np.zeros((0, len(offsets) * feat.shape[1]), dtype=np.float32)
    pad = max(abs(o) for o in offsets)
    padded = np.pad(feat, ((pad, pad), (0, 0)), mode="edge")
    cols = [padded[pad + o: pad + o + T] for o in offsets]
    return np.concatenate(cols, axis=1).astype(np.float32)


def frame_features(x: np.ndarray) -> np.ndarray:
    """Waveform -> (T, D) features for the frame classifier (PCEN + stacked context)."""
    return stack_context(pcen(x))

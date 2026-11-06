"""Offline replay of the browser's live-listening loop (frontend/app.js) on long audio streams.

The page asks the server every 0.7 s to analyse the last <= 6 s of microphone audio, accepts a phrase once it is
followed by 0.3 s of quiet, and forgets the buffer after 2 s of silence.  A television, a conversation or a podcast is
therefore not 'one clip per trial' but thousands of overlapping trials per hour: this module counts what that costs in
false wake-ups per hour (and whether a 'Hey Google' embedded in such a stream still triggers).
"""
from __future__ import annotations

import numpy as np

from .config import SR

TICK_S, WINDOW_S, ACCEPT_TAIL_S, REJECT_TAIL_S, RESET_SILENCE_S = 0.7, 6.0, 0.3, 1.2, 2.0


def _rms_peak(x):
    n = len(x) // 320
    return 0.0 if n == 0 else float(np.sqrt((x[:n * 320].reshape(n, 320).astype(np.float64) ** 2).mean(axis=1)).max())


def run_stream(det, x: np.ndarray, live: bool = True, confirm: bool = False, hysteresis: bool = False):
    """Return [(time_s, score)] of every accepted wake-up while listening to x."""
    x = np.asarray(x, np.float32)
    clear_at, last_end, events, pending = 0, -10.0, [], None
    t = TICK_S
    while t * SR <= len(x):
        end = int(t * SR)
        start = max(clear_at, end - int(WINDOW_S * SR))
        buf = x[start:end]
        t += TICK_S
        if len(buf) < 0.8 * SR:
            continue
        peak = _rms_peak(buf[-4 * SR:])
        if peak < 0.004:
            continue
        tailn = int(RESET_SILENCE_S * SR)
        if len(buf) > tailn + 0.3 * SR and _rms_peak(buf[-tailn:]) < max(0.004, 0.12 * peak):
            clear_at = end
            continue
        res = det.analyze(buf, repair=False, with_curves=False, live=live)
        if not res.get("span"):
            continue
        abs_end = start / SR + res["span"]["end"]
        tail = res.get("tail_silence_s")
        fresh = abs_end > last_end + 0.5
        if confirm and pending is not None:
            ok = (res["detected"] or (hysteresis and res.get("soft"))) and abs(abs_end - pending[0]) < 0.4
            sc = pending[1] if hysteresis else (min(pending[1], res["score"]) if ok else None)
            sc = sc if ok else None
            pending = None
            if ok:                       # the same phrase was found again one tick later: accept
                events.append((end / SR, sc))
                last_end, clear_at = abs_end, end
            continue
        if res["detected"] and tail is not None and tail >= ACCEPT_TAIL_S and fresh:
            if confirm:
                pending = (abs_end, res["score"])
                continue
            events.append((end / SR, res["score"]))
            last_end, clear_at = abs_end, end
        elif (not res["detected"]) and tail is not None and tail >= REJECT_TAIL_S and fresh:
            last_end = abs_end
    return events


def make_stream(clips, rng, minutes: float, gap_range=(0.05, 0.6), room: bool = True, level: float = 0.06):
    """Continuous talk: random clips back to back with short pauses, optional room / noise channel per 30 s block."""
    from .augment import augment_channel
    out, total = [], 0
    while total < minutes * 60 * SR:
        c = clips[int(rng.integers(0, len(clips)))]
        c = c / (np.abs(c).max() + 1e-9) * 0.5
        out += [c, np.zeros(int(rng.uniform(*gap_range) * SR), np.float32)]
        total += len(out[-1]) + len(c)
    x = np.concatenate(out)[: int(minutes * 60 * SR)]
    if room:
        blocks = []
        for i in range(0, len(x), 30 * SR):
            b, _ = augment_channel(x[i:i + 30 * SR], rng, strength=1.0)
            blocks.append(b)
        x = np.concatenate(blocks)
    return (x / (np.sqrt((x.astype(np.float64) ** 2).mean()) + 1e-9) * level).astype(np.float32)


def embed(wake, rng, lead_s=3.0, tail_s=3.0, level_ref=None):
    """A wake-word clip surrounded by quiet room noise (what a person at a quiet desk produces)."""
    noise = lambda n: (rng.standard_normal(n) * 0.0015).astype(np.float32)
    w = wake / (np.abs(wake).max() + 1e-9) * 0.4
    return np.concatenate([noise(int(lead_s * SR)), w, noise(int(tail_s * SR))])

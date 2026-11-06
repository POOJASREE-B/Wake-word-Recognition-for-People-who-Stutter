"""Evaluation at a fixed false-accept RATE (false wake-ups per hour of negative audio), not a clip-level FAR.

A product wake word is specified as "FRR at <= X false wake-ups / hour".  This module
  * builds long negative streams from any folders of audio (speech, noise, music, TV ...),
  * replays them through the exact live loop of the page (livesim.run_stream, confirm rule on) and keeps every
    candidate wake-up with its score,
  * turns the candidates into FA/hr(threshold) with exact Poisson 95% intervals,
  * scores positives and gives FRR(threshold) with Wilson intervals, per disfluency type,
  * writes the DET table (FRR at fixed FA/hr values) honestly: with H hours of negatives the smallest rate that can be
    resolved is 1/H per hour; zero events only bound the rate by 3/H (rule of three, 95%).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import livesim
from .config import SR

AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3", ".m4a")


# ----------------------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def poisson_ci(k, hours, conf=0.95):
    """Exact (Garwood) interval for an event rate (per hour)."""
    from math import exp, lgamma, log
    a = 1 - conf

    def cdf(kk, lam):                                # P(X <= kk)
        return sum(exp(-lam + i * log(lam) - lgamma(i + 1)) for i in range(kk + 1)) if lam > 0 else 1.0

    lo = 0.0
    if k > 0:
        l, h = 0.0, 10.0 * (k + 5)
        for _ in range(80):
            m = (l + h) / 2
            (h, l) = (m, l) if 1 - cdf(k - 1, m) > a / 2 else (h, m)
        lo = (l + h) / 2
    l, h = 0.0, 10.0 * (k + 5)
    for _ in range(80):
        m = (l + h) / 2
        (h, l) = (h, m) if cdf(k, m) > a / 2 else (m, l)
    return (lo / hours, ((l + h) / 2) / hours)


# ----------------------------------------------------------------------------- audio sources
def load_audio(path):
    from .audio_io import load_audio as _l
    return np.asarray(_l(path), np.float32)


def list_audio(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in AUDIO_EXT)


def make_pool_streams(files, rng, minutes=12.0, level_db=(-30.0, -14.0), gap_s=(0.0, 1.5), loader=load_audio):
    """Concatenate audio files (random order, random level, short gaps) into streams of ~`minutes`; yields np arrays.
    Every file is used once; the last stream may be shorter."""
    order = rng.permutation(len(files))
    buf, n = [], 0
    for i in order:
        try:
            x = loader(files[i])
        except Exception:
            continue
        if len(x) < 0.3 * SR:
            continue
        pk = float(np.max(np.abs(x))) + 1e-9
        g = 10 ** (rng.uniform(*level_db) / 20) / pk
        buf.append((x * g).astype(np.float32))
        buf.append(np.zeros(int(rng.uniform(*gap_s) * SR), np.float32) + rng.normal(0, 1e-4, 1).astype(np.float32))
        n += len(buf[-2]) + len(buf[-1])
        if n >= minutes * 60 * SR:
            yield np.concatenate(buf)
            buf, n = [], 0
    if n > 20 * SR:
        yield np.concatenate(buf)


def candidates(det, streams, thr_floor=0.2, confirm=True, log=print, name=""):
    """Run the live loop over `streams` with a low threshold; return (hours, [score of every candidate wake-up])."""
    old = det.verifier.threshold
    det.verifier.threshold = thr_floor
    sc, secs = [], 0.0
    try:
        for i, x in enumerate(streams):
            ev = livesim.run_stream(det, x, live=False, confirm=confirm)
            sc += [float(s) for _, s in ev]
            secs += len(x) / SR
            log(f"      {name} stream {i + 1}: {len(x) / SR / 60:.1f} min, {len(ev)} candidates")
    finally:
        det.verifier.threshold = old
    return secs / 3600.0, sc


def far_table(cands_by_pool, thresholds):
    """{pool: (hours, scores)} -> rows of FA/hr per threshold per pool and pooled, with exact 95% CIs."""
    rows = []
    for t in thresholds:
        row = {"threshold": float(t)}
        tot_k, tot_h = 0, 0.0
        for name, (h, sc) in cands_by_pool.items():
            k = int(sum(s >= t for s in sc))
            row[name] = {"events": k, "hours": round(h, 3), "per_hour": k / h if h else 0.0, "ci95": poisson_ci(k, h) if h else (0, 0)}
            tot_k += k
            tot_h += h
        row["pooled"] = {"events": tot_k, "hours": round(tot_h, 3), "per_hour": tot_k / tot_h if tot_h else 0.0,
                         "ci95": poisson_ci(tot_k, tot_h) if tot_h else (0, 0)}
        rows.append(row)
    return rows


def frr_at(pos_scores, thr):
    pos_scores = np.asarray(pos_scores, float)
    miss = int((pos_scores < thr).sum())
    return miss / max(len(pos_scores), 1), wilson(miss, len(pos_scores))


def det_table(pos_by_group, cands_by_pool, targets=(0.2, 0.5, 1.0, 2.0, 5.0), grid=None):
    """For every target FA/hr: the lowest threshold whose pooled FA/hr is <= target, and FRR per positive group there.
    `pos_by_group` {name: scores}. Also reports the resolution limit of the negative pool."""
    grid = np.round(np.arange(0.2, 0.96, 0.01), 2) if grid is None else grid
    far = far_table(cands_by_pool, grid)
    hours = far[0]["pooled"]["hours"]
    out = {"negative_hours": hours, "resolution_per_hour": (1.0 / hours if hours else None),
           "rule_of_three_upper_per_hour": (3.0 / hours if hours else None), "rows": []}
    for tgt in targets:
        ok = [r for r in far if r["pooled"]["per_hour"] <= tgt]
        if not ok:
            out["rows"].append({"target_fa_per_hour": tgt, "threshold": None})
            continue
        r = ok[0]
        row = {"target_fa_per_hour": tgt, "threshold": r["threshold"], "fa_per_hour": r["pooled"]["per_hour"],
               "fa_ci95": r["pooled"]["ci95"], "frr": {}}
        for g, sc in pos_by_group.items():
            f, ci = frr_at(sc, r["threshold"])
            row["frr"][g] = {"frr": f, "ci95": ci, "n": len(sc)}
        out["rows"].append(row)
    out["far_grid"] = far
    return out


def det_markdown(det, title="", min_n=1):
    L = [f"### {title}", "",
         f"Negative audio: **{det['negative_hours']:.2f} h**. The smallest false-accept rate this pool can resolve is "
         f"{det['resolution_per_hour']:.2f}/h; zero events would only bound the rate below {det['rule_of_three_upper_per_hour']:.2f}/h (95%).", ""]
    groups = sorted({g for r in det["rows"] if r.get("frr") for g in r["frr"]})
    L += ["| target FA/h | threshold | measured FA/h (95% CI) | " + " | ".join(f"FRR {g}" for g in groups) + " |",
          "|---|---|---|" + "---|" * len(groups)]
    for r in det["rows"]:
        if r["threshold"] is None:
            L.append(f"| {r['target_fa_per_hour']} | not reachable | | " + " | ".join("" for _ in groups) + " |")
            continue
        cells = []
        for g in groups:
            c = r["frr"].get(g)
            cells.append(f"{c['frr']:.2f} ({c['ci95'][0]:.2f}-{c['ci95'][1]:.2f}), n={c['n']}" if c else "")
        L.append(f"| {r['target_fa_per_hour']} | {r['threshold']:.2f} | {r['fa_per_hour']:.2f} ({r['fa_ci95'][0]:.2f}-{r['fa_ci95'][1]:.2f}) | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def save(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


# ----------------------------------------------------------------------------- hard-negative mining
def mine(det, streams, thr_floor=0.35, pre_s=4.5, log=print, name=""):
    """Replay negative audio through the live loop and keep the audio just before every candidate wake-up.
    Returns [(waveform, score)].  The audio must NOT contain the wake word (it is added as a negative)."""
    old = det.verifier.threshold
    det.verifier.threshold = thr_floor
    out = []
    try:
        for i, x in enumerate(streams):
            ev = livesim.run_stream(det, x, live=False, confirm=True)
            for t, s in ev:
                a, b = max(0, int((t - pre_s) * SR)), min(len(x), int(t * SR))
                if b - a > 0.8 * SR:
                    out.append((np.asarray(x[a:b], np.float32).copy(), float(s)))
            log(f"      {name} stream {i + 1}: {len(ev)} false candidates mined")
    finally:
        det.verifier.threshold = old
    return out

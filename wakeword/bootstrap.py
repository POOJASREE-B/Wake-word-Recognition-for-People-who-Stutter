"""Add the raw WhatsApp recordings that are not yet part of the curated set.

Raw recordings are messy: several attempts, extra words, long silences. For every recording the
candidate 'Hey Google' windows are cut out, aligned with the first-pass model and scored with the
detector; a recording is only used if one window is accepted. Everything (accepted / rejected,
score, window) is written to source_audit.csv so it can be reviewed by listening.
"""
from __future__ import annotations

import numpy as np

from . import sources
from .align import refine_utterance
from .config import HOP, SR, WIN
from .detector import WakeWordDetector
from .segmenter import build_utterance, voiced_regions
from .sources import Source, _unit_sane, speech_qa


def _candidate_windows(x, max_len=2.6, min_len=0.5):
    regs, _ = voiced_regions(x)
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
            if min_len <= (b - a) / 100.0 <= max_len:
                cands.add((a, b))
    for i in range(len(regs)):
        for j in range(i, min(len(regs), i + 4)):
            a, b = regs[i][0], regs[j][1]
            if min_len <= (b - a) / 100.0 <= max_len:
                cands.add((a, b))
    pad = int(0.04 * SR)
    return [(max(0, a * HOP - pad), min(len(x), b * HOP + WIN + pad)) for a, b in sorted(cands)]


def verify_new_recordings(detector: WakeWordDetector, raw_dir, skip_names, min_score: float | None = None, log=print):
    """Pick the best 'Hey Google' window inside every raw recording.

    The recordings are labelled 'Hey Google' by their authors, so the model is only used to RANK the
    candidate windows of a recording (and forced alignment must give plausible Hey / Goo / gle units).
    min_score=None accepts the best window whatever its score; the score is written to the audit file
    so doubtful recordings can be reviewed by listening.
    Returns (accepted_sources, audit_rows).
    """
    items = sources.load_raw_recordings(raw_dir, skip_names)
    accepted, audit = [], []
    for k, (name, x) in enumerate(items):
        ranked = []
        for s, e in _candidate_windows(x):
            w = x[max(0, s - int(0.15 * SR)): min(len(x), e + int(0.15 * SR))]
            ranked.append((detector.score_wave(w), s, e))
        ranked.sort(reverse=True)
        sid = f"NEW_{k + 1:03d}"
        row = {"source": sid, "group": sid, "kind": "real", "origin": name, "status": "REJECTED_no_voiced_window"}
        for sc, s, e in ranked[:12]:                       # best window whose units align plausibly
            if min_score is not None and sc < min_score:
                row["status"] = f"REJECTED_low_score({sc:.2f})"
                break
            u = build_utterance(x, (s, e), meta={"origin": name})
            u2 = refine_utterance(detector.model, u) if u is not None else None
            if u2 is not None and _unit_sane(u2) and speech_qa(u2)[0]:
                accepted.append(Source(id=sid, group=sid, kind="real", utt=u2, origin=name))
                row["status"] = f"used(score={sc:.2f},window={s / SR:.2f}-{e / SR:.2f}s,UNREVIEWED)"
                break
            # every recording in this folder IS a fluent 'Hey Google' (stated by the authors): when the model-based
            # alignment fails, keep the energy-based Hey / Goo / gle boundaries instead of dropping a real voice
            if u is not None and _unit_sane(u) and speech_qa(u)[0]:
                accepted.append(Source(id=sid, group=sid, kind="real", utt=u, origin=name))
                row["status"] = f"used_energy_boundaries(score={sc:.2f},window={s / SR:.2f}-{e / SR:.2f}s)"
                break
            row["status"] = f"REJECTED_bad_boundaries(score={sc:.2f})"
        audit.append(row)
    log(f"      raw recordings: {len(accepted)}/{len(items)} usable (best window per recording; see source_audit.csv)")
    return accepted, audit

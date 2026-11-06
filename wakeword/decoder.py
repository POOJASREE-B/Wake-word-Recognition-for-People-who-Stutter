"""Disfluency-tolerant keyword decoder.

The acoustic model gives, for every 10 ms frame, the posterior of  bg | hey | goo | gle | other.
A small finite-state grammar then searches for the phrase

    garbage*  HEY+ ( gap HEY+ )*  ( gap )?  GOO+ ( gap GOO+ )*  ( gap )?  GLE+  ( gap GOO+ ... GLE+ )*  garbage*

so that *stuttered* realisations are accepted:
    sound / syllable repetition   H-H-Hey, Goo-goo-Google   -> HEY / GOO re-entry through a gap
    prolongation                  Heeeey, Gooooogle        -> self-loops on HEY / GOO
    block                         Hey ........ Google      -> long BG gap between units
    word repetition               Hey hey Google, Google Google
while the keyword path still *requires* Hey -> Goo -> gle in that order, so unrelated speech,
only 'Hey', only 'Google', 'Google Hey', noise etc. cannot be explained by it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BG, FILLER_FRAME_PENALTY, GLE, GOO, HEY, MIN_FRAMES, OTHER, UNIT_GATE

NEG = -1e9
RESTART_PENALTY = -1.2
GAP_PENALTY = -0.4


@dataclass
class Grammar:
    names: list
    emit: list              # class index, or -1 for 'garbage' (bg/other)
    logA: np.ndarray
    start: np.ndarray
    end: np.ndarray


def build_grammar(mh=MIN_FRAMES[HEY], mg=MIN_FRAMES[GOO], ml=MIN_FRAMES[GLE], strict: bool = False) -> Grammar:
    names, emit = [], []

    def add(kind, cls, n=1):
        ids = []
        for i in range(n):
            names.append((kind, i))
            emit.append(cls)
            ids.append(len(names) - 1)
        return ids

    pre = add("PRE", -1)[0]
    H = add("H", HEY, mh)
    HG = add("HG", -2)[0]
    G = add("G", GOO, mg)
    GG = add("GG", -2)[0]
    L = add("L", GLE, ml)
    LG = add("LG", -2)[0]
    post = add("POST", -1)[0]
    N = len(names)
    A = np.full((N, N), NEG, dtype=np.float32)

    def link(a, b, w=0.0):
        A[a, b] = max(A[a, b], w)

    link(pre, pre); link(pre, H[0])
    for ch in (H, G, L):
        for i in range(len(ch) - 1):
            link(ch[i], ch[i + 1])
        link(ch[-1], ch[-1])
    link(H[-1], G[0]); link(G[-1], L[0]); link(L[-1], post); link(post, post)
    if not strict:                       # disfluency paths: gaps (blocks) and restarts (repetitions)
        link(H[-1], HG, GAP_PENALTY)
        link(HG, HG); link(HG, H[0], RESTART_PENALTY); link(HG, G[0])
        link(G[-1], GG, GAP_PENALTY); link(G[-1], G[0], RESTART_PENALTY)
        link(GG, GG); link(GG, G[0], RESTART_PENALTY); link(GG, L[0])
        link(GG, H[0], RESTART_PENALTY)  # revision / restart: 'Hey Goo- ... Hey Google'
        link(L[-1], LG, GAP_PENALTY)
        link(LG, LG); link(LG, G[0], RESTART_PENALTY); link(LG, post)
    start = np.full(N, NEG, dtype=np.float32)
    start[pre] = 0.0
    start[H[0]] = 0.0
    end = np.full(N, NEG, dtype=np.float32)
    end[[post, L[-1], LG]] = 0.0
    return Grammar(names, emit, A, start, end)


_GRAMMAR = build_grammar()
_STRICT = build_grammar(strict=True)


def _emissions(logp: np.ndarray, g: Grammar) -> np.ndarray:
    T = logp.shape[0]
    E = np.zeros((T, len(g.names)), dtype=np.float32)
    garb = np.maximum(logp[:, BG], logp[:, OTHER])
    gap = np.maximum(logp[:, BG], logp[:, OTHER] + FILLER_FRAME_PENALTY)     # pause or filled pause (um / uh)
    for s, c in enumerate(g.emit):
        E[:, s] = garb if c == -1 else (gap if c == -2 else logp[:, c])
    return E


def viterbi(logp: np.ndarray, g: Grammar = _GRAMMAR):
    """Best grammar path. Returns (score, state_path) or (None, None) if no valid path."""
    T, N = logp.shape[0], len(g.names)
    E = _emissions(logp, g)
    delta = np.full((T, N), NEG, dtype=np.float32)
    psi = np.zeros((T, N), dtype=np.int16)
    delta[0] = g.start + E[0]
    for t in range(1, T):
        cand = delta[t - 1][:, None] + g.logA            # (from, to)
        psi[t] = np.argmax(cand, axis=0)
        delta[t] = cand[psi[t], np.arange(N)] + E[t]
    final = delta[-1] + g.end
    s = int(np.argmax(final))
    if final[s] < NEG / 2:
        return None, None
    path = np.zeros(T, dtype=np.int16)
    path[-1] = s
    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return float(final[s]), path


FEATURE_NAMES = ["deficit_per_frame", "deficit_total", "span_s", "mean_lp_hey", "mean_lp_goo", "mean_lp_gle",
                 "min_mean_lp", "n_hey", "n_goo", "n_gle", "restarts_hey", "restarts_goo", "gap_total_s",
                 "gap_max_s", "frac_other_in_span", "max_lp_hey", "max_lp_goo", "max_lp_gle", "min_max_lp",
                 # v1.2d: STRUCTURE-FREE evidence - how well the unit frames themselves match, ignoring the grammar's
                 # restart / pause / filler penalties (which grow with the amount of stuttering, not with mismatch)
                 "unit_deficit_pf", "top5_hey", "top5_goo", "top5_gle", "min_top5", "gap_quality"]


def _structure_free(logp, kinds, a, b, h_idx, g_idx, l_idx):
    """Evidence that does not depend on HOW MANY times a unit was repeated or how long the pauses were:
    unit_deficit_pf = mean shortfall of the unit frames vs the best class at that frame (no transition penalties),
    top5_<unit>     = mean of the 5 most confident frames of each unit (a short 'g-' fragment does not dilute it),
    gap_quality     = how well the pause frames inside the phrase look like silence or a filled pause."""
    best = logp.max(axis=1)
    unit = np.concatenate([h_idx, g_idx, l_idx])
    cls = np.concatenate([np.full(len(h_idx), HEY), np.full(len(g_idx), GOO), np.full(len(l_idx), GLE)])
    unit_def = float((best[unit] - logp[unit, cls]).mean()) if unit.size else 0.0
    top = lambda ids, c: float(np.sort(logp[ids, c])[-min(5, len(ids)):].mean())
    t_h, t_g, t_l = top(h_idx, HEY), top(g_idx, GOO), top(l_idx, GLE)
    gaps = np.where(np.isin(kinds[a:b], ["HG", "GG", "LG"]))[0] + a
    gq = float(np.maximum(logp[gaps, BG], logp[gaps, OTHER]).mean()) if gaps.size else 0.0
    return [unit_def, t_h, t_g, t_l, min(t_h, t_g, t_l), gq]


def _runs(path_kind: np.ndarray):
    runs, s = [], 0
    for i in range(1, len(path_kind) + 1):
        if i == len(path_kind) or path_kind[i] != path_kind[s]:
            runs.append((str(path_kind[s]), s, i))
            s = i
    return runs


def decode(logp: np.ndarray, gate: float | None = UNIT_GATE):
    """Run the grammar and compute the clip-level evidence features.

    Returns dict(features=np.ndarray|None, runs=[(kind,start,end)], span=(a,b)|None, deficit=float)
    """
    T = logp.shape[0]
    empty = {"features": None, "runs": [], "span": None, "deficit": float("inf"), "path": None}
    if T < 12:
        return empty
    score, path = viterbi(logp)
    if path is None:
        return empty
    kinds = np.array([_GRAMMAR.names[s][0] for s in path])
    free = float(logp.max(axis=1).sum())
    deficit = free - score
    runs = _runs(kinds)
    idx = np.where(np.isin(kinds, ["H", "HG", "G", "GG", "L", "LG"]))[0]
    h_idx, g_idx, l_idx = (np.where(kinds == k)[0] for k in ("H", "G", "L"))
    if idx.size == 0 or h_idx.size == 0 or g_idx.size == 0 or l_idx.size == 0:
        return {**empty, "runs": runs, "deficit": deficit, "path": kinds}
    a = int(h_idx.min())
    b = int(l_idx.max()) + 1
    # trailing / leading gap states inside the span are not part of the phrase
    span = b - a
    gap_runs = [(k, s, e) for k, s, e in runs if k in ("HG", "GG", "LG") and s >= a and e <= b]
    gap_total = sum(e - s for _, s, e in gap_runs) / 100.0
    gap_max = max([e - s for _, s, e in gap_runs] + [0]) / 100.0
    # deficit restricted to the span (the pre/post garbage frames carry ~0 deficit by construction)
    mean_lp = lambda ids, c: float(logp[ids, c].mean())
    am = np.argmax(logp[a:b], axis=1)
    restarts_h = sum(1 for k, s, e in runs if k == "H" and s > a)
    restarts_g = sum(1 for k, s, e in runs if k == "G") - 1
    max_h, max_g, max_l = (float(logp[h_idx, HEY].max()), float(logp[g_idx, GOO].max()), float(logp[l_idx, GLE].max()))
    if gate and min(max_h, max_g, max_l) < np.log(gate):
        # one of Hey / Goo / gle was never clearly heard: the path was forced through it -> not the wake word
        weak = [n for n, v in (("Hey", max_h), ("Goo", max_g), ("gle", max_l)) if v < np.log(gate)]
        return {**empty, "runs": runs, "deficit": deficit, "path": kinds, "gated": weak}
    feats = np.array([
        deficit / max(span, 1), np.log1p(max(deficit, 0.0)), span / 100.0,
        mean_lp(h_idx, HEY), mean_lp(g_idx, GOO), mean_lp(l_idx, GLE),
        min(mean_lp(h_idx, HEY), mean_lp(g_idx, GOO), mean_lp(l_idx, GLE)),
        len(h_idx) / 100.0, len(g_idx) / 100.0, len(l_idx) / 100.0,
        restarts_h, max(restarts_g, 0), gap_total, gap_max,
        float((am == OTHER).mean()), max_h, max_g, max_l, min(max_h, max_g, max_l),
        *_structure_free(logp, kinds, a, b, h_idx, g_idx, l_idx)], dtype=np.float32)
    return {"features": feats, "runs": runs, "span": (a, b), "deficit": deficit, "path": kinds}

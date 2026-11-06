"""Why was (or wasn't) a clip accepted?  python run.py diagnose FILE_OR_FOLDER   |   python run.py diagnose --team

For every clip: decision, verifier score, threshold, the peak frame posterior of Hey / Goo / gle on the decoded path
(the unit gate needs each >= the gate), and the reason for a rejection.  --team runs the model over the stuttered
and fluent clips of the two project datasets and summarises the detection rate per disfluency type."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import audio_io, decoder
from .config import RAW_DIR

AUDIO = (".wav", ".mp3", ".ogg", ".m4a", ".aac", ".mpeg", ".mp4", ".flac", ".opus", ".webm")
I_H, I_G, I_L = (decoder.FEATURE_NAMES.index(n) for n in ("max_lp_hey", "max_lp_goo", "max_lp_gle"))


def explain(det, x) -> dict:
    lp = det.model.predict_wave(x)
    raw = decoder.decode(lp, None)                       # without the gate, to see the unit peaks
    res = det.analyze(x, repair=False, with_curves=False)
    out = {"detected": bool(res["detected"]), "score": float(res["score"]), "threshold": det.threshold,
           "gate": det.gate, "heard": res.get("heard_text", ""), "peaks": None}
    f = raw["features"]
    if f is None:
        out["reason"] = "no Hey-Goo-gle path"
    else:
        out["peaks"] = tuple(round(float(np.exp(f[i])), 2) for i in (I_H, I_G, I_L))
        weak = [n for n, p in zip(("Hey", "Goo", "gle"), out["peaks"]) if p < det.gate]
        if out["detected"]:
            out["reason"] = "accepted"
        elif weak:
            out["reason"] = "unit not clearly heard: " + ",".join(weak)
        else:
            out["reason"] = "score below threshold"
    return out


def _line(name, e):
    pk = "-" if e["peaks"] is None else "/".join(f"{p:.2f}" for p in e["peaks"])
    return (f"{'YES' if e['detected'] else 'no ':3s}  score={e['score']:.3f}  peaks Hey/Goo/gle={pk:14s}  "
            f"{e['reason']:32s} {name}")


def run(det, target=None, team=False, raw_dir=RAW_DIR, log=print):
    log(f"model: {(det.meta or {}).get('name', '?')}   threshold={det.threshold:.3f}   unit gate={det.gate:.2f}")
    if team:
        from .sources import load_team_disfluent
        stats = defaultdict(Counter)
        reasons = defaultdict(Counter)
        for t in load_team_disfluent(Path(raw_dir), log=log):
            e = explain(det, t["x"])
            k = f"{'real voice' if t['kind'] == 'real' else 'TTS voice '} | {t['dtype']}"
            stats[k]["n"] += 1
            stats[k]["hit"] += int(e["detected"])
            reasons[k][e["reason"].split(":")[0]] += 1
        log("\nrecorded-style stuttered clips of the project datasets (all of them, any split):")
        log(f"{'group':42s} {'n':>4s} {'detected':>9s}   main reasons for a miss")
        for k in sorted(stats):
            s = stats[k]
            miss = ", ".join(f"{r} {c}" for r, c in reasons[k].most_common() if r != "accepted")
            log(f"{k:42s} {s['n']:4d} {s['hit'] / max(s['n'], 1):9.1%}   {miss}")
        return
    p = Path(target)
    files = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.suffix.lower() in AUDIO)
    reasons = Counter()
    hits = 0
    for q in files:
        try:
            x = audio_io.load_audio(q)
        except Exception as exc:
            log(f"ERR  cannot decode {q.name}: {exc}")
            continue
        e = explain(det, x)
        hits += int(e["detected"])
        reasons[e["reason"].split(":")[0]] += 1
        log(_line(q.name, e))
    if len(files) > 1:
        log(f"\n{hits}/{len(files)} detected.  reasons: " + ", ".join(f"{r} {c}" for r, c in reasons.most_common()))

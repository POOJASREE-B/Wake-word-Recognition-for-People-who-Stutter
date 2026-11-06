"""Honest evaluation: FRR per disfluency type, FAR per negative category, false accepts per hour."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import decoder
from .config import DATA_DIR, MODEL_DIR
from .detector import WakeWordDetector
from .model import load_sample, read_annotations, read_manifest


def evaluate(detector: WakeWordDetector, data_dir=DATA_DIR, split="test", out_dir=MODEL_DIR, log=print) -> dict:
    rows, ann = read_manifest(data_dir), read_annotations(data_dir)
    rows = [r for r in rows if r["split"] == split]
    if not rows:
        raise SystemExit(f"no clips in split '{split}'")
    thr = detector.threshold
    recs = []
    for i, r in enumerate(rows):
        x, _ = load_sample(data_dir, r, ann)
        res = detector.analyze(x, repair=False, with_curves=False)
        recs.append({"id": r["sample_id"], "wake": int(r["is_wake_word"]), "score": res["score"],
                     "hit": bool(res["detected"]), "cat": r["category"], "label": r["label"],
                     "kind": r["speaker_kind"], "dur": float(r["duration_seconds"]), "ctx": r["context"],
                     "word": r["affected_word"], "flag_disf": bool(res["disfluent"]), "n_events": int(r["n_events"]),
                     "profile": r.get("stutter_profile", "n/a"), "emotion": r.get("emotion", "n/a"),
                     "age": r.get("age_group", "n/a"), "sec": r.get("secondary_behaviour", "none"),
                     "method": r.get("generation_method", "")})
        if (i + 1) % 500 == 0:
            log(f"      evaluated {i + 1}/{len(rows)}")
    pos = [r for r in recs if r["wake"]]
    neg = [r for r in recs if not r["wake"]]

    def rate(sub, key):
        return round(float(np.mean([key(r) for r in sub])), 4) if sub else None

    def group(sub, keyfn, val):
        d = defaultdict(list)
        for r in sub:
            d[keyfn(r)].append(r)
        return {k: {"n": len(v), val: rate(v, lambda r: (not r["hit"]) if val == "FRR" else r["hit"])} for k, v in sorted(d.items())}

    neg_hours = sum(r["dur"] for r in neg) / 3600.0
    fa = sum(r["hit"] for r in neg)
    out = {
        "split": split, "threshold": round(thr, 4), "n_wake_clips": len(pos), "n_negative_clips": len(neg),
        "FRR_overall": rate(pos, lambda r: not r["hit"]),
        "FRR_fluent": rate([r for r in pos if r["label"] == "fluent"], lambda r: not r["hit"]),
        "FRR_disfluent": rate([r for r in pos if r["label"] == "disfluent"], lambda r: not r["hit"]),
        "FRR_by_disfluency_type": group(pos, lambda r: r["cat"], "FRR"),
        "FRR_by_speaker_kind": group(pos, lambda r: r["kind"], "FRR"),
        "FRR_by_context": group(pos, lambda r: r["ctx"], "FRR"),
        "FRR_by_stutter_profile": group(pos, lambda r: r["profile"], "FRR"),
        "FRR_by_emotion": group(pos, lambda r: r["emotion"], "FRR"),
        "FRR_by_age_group": group(pos, lambda r: r["age"], "FRR"),
        "FRR_by_secondary_behaviour": group(pos, lambda r: r["sec"], "FRR"),
        # how the wake word was produced x who spoke it: separates our generator's stutters from the project's
        # recorded-style ones, and real voices from TTS voices
        "FRR_by_source": group(pos, lambda r: f"{r['label']} / {r['method']} / {r['kind']}", "FRR"),
        "FAR_overall": rate(neg, lambda r: r["hit"]),
        "false_accepts": int(fa),
        "false_accepts_per_hour": round(fa / max(neg_hours, 1e-9), 2),
        "negative_hours": round(neg_hours, 3),
        "FAR_by_negative_category": group(neg, lambda r: r["cat"], "FAR"),
        "disfluency_flag_on_disfluent_hits": rate([r for r in pos if r["label"] == "disfluent" and r["hit"]], lambda r: r["flag_disf"]),
        "disfluency_flag_on_fluent_hits": rate([r for r in pos if r["label"] == "fluent" and r["hit"]], lambda r: r["flag_disf"]),
    }
    try:
        from sklearn.metrics import roc_auc_score
        out["AUC"] = round(float(roc_auc_score([r["wake"] for r in recs], [r["score"] for r in recs])), 4)
    except Exception:
        out["AUC"] = None
    table = []
    for t in (0.3, 0.5, 0.7, 0.9, 0.95, 0.99):
        table.append({"threshold": t, "FRR": round(float(np.mean([r["score"] < t for r in pos])), 4),
                      "FAR": round(float(np.mean([r["score"] >= t for r in neg])), 4)})
    out["threshold_table"] = table
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{split}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (out_dir / f"report_{split}.md").write_text(_markdown(out), encoding="utf-8")
    log(_markdown(out).encode("ascii", "replace").decode("ascii"))
    return out


def _markdown(m) -> str:
    L = [f"# Evaluation on the **{m['split']}** split (threshold {m['threshold']})", "",
         f"- wake-word clips: {m['n_wake_clips']}   negative clips: {m['n_negative_clips']} ({m['negative_hours']} h)",
         f"- **FRR** (missed wake words): overall {m['FRR_overall']}  | fluent {m['FRR_fluent']}  | disfluent {m['FRR_disfluent']}",
         f"- **FAR** (false accepts on negatives): {m['FAR_overall']}  ({m['false_accepts']} clips, "
         f"{m['false_accepts_per_hour']} per hour of negative audio)   AUC: {m['AUC']}", "",
         "## FRR by disfluency type", "", "| type | n | FRR |", "|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['FRR']} |" for k, v in m["FRR_by_disfluency_type"].items()]
    L += ["", "## FAR by negative category", "", "| category | n | FAR |", "|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['FAR']} |" for k, v in m["FAR_by_negative_category"].items()]
    L += ["", "## FRR by speaker kind", "", "| kind | n | FRR |", "|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['FRR']} |" for k, v in m["FRR_by_speaker_kind"].items()]
    for key, title in (("FRR_by_stutter_profile", "stuttering profile"), ("FRR_by_emotion", "emotion"),
                       ("FRR_by_age_group", "age group"), ("FRR_by_secondary_behaviour", "secondary behaviour"),
                       ("FRR_by_source", "source (label / how it was made / voice)")):
        if m.get(key):
            L += ["", f"## FRR by {title}", "", f"| {title} | n | FRR |", "|---|---|---|"]
            L += [f"| {k} | {v['n']} | {v['FRR']} |" for k, v in m[key].items()]
    L += ["", "## Threshold sweep", "", "| threshold | FRR | FAR |", "|---|---|---|"]
    L += [f"| {t['threshold']} | {t['FRR']} | {t['FAR']} |" for t in m["threshold_table"]]
    return "\n".join(L) + "\n"

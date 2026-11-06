#!/usr/bin/env python3
"""Study the stuttering in ALL real recordings (every phrase), preprocess them, and write a stuttering dataset.

    python scripts/pattern_study.py --real real_people --out stutter_top --variants 12

Outputs <out>/clean_pp/*.wav (preprocessed real clips), <out>/annotations.json (measured pattern per clip),
<out>/pattern_report.md, <out>/synthetic/*.wav + <out>/manifest.csv (stuttered variants of the real clips; split by SPEAKER).
Every number comes from the acoustic detectors in wakeword/patterns.py; their recall is limited (see report), so the
counts are lower bounds, not ground truth."""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wakeword import patterns as P, realpeople as rp, robust
from wakeword.config import SR
from wakeword.audio_io import save_wav


def preprocess(x):
    """DC removal, 70 Hz high-pass (rumble/hum), speech-level normalisation to -20 dBFS RMS over voiced frames, peak limit."""
    x = np.asarray(x, np.float64); x = x - x.mean()
    a = np.exp(-2 * np.pi * 70 / SR)                       # one-pole high-pass, applied twice
    for _ in range(2):
        y = np.empty_like(x); p_in = p_out = 0.0
        for i, v in enumerate(x):
            p_out = a * (p_out + v - p_in); p_in = v; y[i] = p_out
        x = y
    segs = robust.voiced_segments(x.astype(np.float32))
    ref = np.concatenate([x[a_:b_] for a_, b_ in segs]) if segs else x
    g = 10 ** (-20 / 20) / (np.sqrt((ref ** 2).mean()) + 1e-9)
    x = x * g
    pk = np.abs(x).max()
    if pk > 0.95: x = x * 0.95 / pk
    return x.astype(np.float32)


def annotate(x):
    e = P.acoustic_events(x); st = P.clip_stats(x); rs = P.restarts(x)
    return {"span_s": None if st is None else round(float(st[0]), 3), "n_voiced": e["n_seg"], "speech_s": round(e["speech_s"], 3),
            "total_gap_s": None if st is None else round(float(st[2]), 3), "max_gap_s": None if st is None else round(float(st[3]), 3),
            "blocks_s": e["blocks"], "pauses_s": e["pauses"], "repetitions": e["reps"], "repetition_runs": e["rep_runs"],
            "prolongations_s": e["prolong"], "restarts": [{"from_s": round(r[3], 2), "to_s": round(r[4], 2), "dtw": round(r[2], 3)} for r in rs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="real_people"); ap.add_argument("--out", default="stutter_top")
    ap.add_argument("--variants", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pattern", default="", help="PatternModel json to sample from (default: fitted file if present, else defaults)")
    a = ap.parse_args()
    out = Path(a.out); (out / "clean_pp").mkdir(parents=True, exist_ok=True); (out / "synthetic").mkdir(exist_ok=True)
    M = rp.load_manifest(a.real); rng = np.random.default_rng(a.seed)
    model = P.PatternModel.from_json(Path(a.pattern).read_text()) if a.pattern and Path(a.pattern).exists() else P.PatternModel()
    ann, clips = {}, {}
    for m in M:
        x = preprocess(rp.load_clip(m, a.real)); clips[m["sample_id"]] = x
        save_wav(out / "clean_pp" / f"{m['sample_id']}.wav", x)
        ann[m["sample_id"]] = {"speaker": m["speaker"], "phrase": m["phrase"], **annotate(x)}
    (out / "annotations.json").write_text(json.dumps(ann, indent=1))
    # ---- aggregate
    by = defaultdict(list)
    for k, v in ann.items(): by[v["phrase"]].append(v)
    L = ["# Stuttering pattern of the real recordings", "", f"{len(ann)} clips, {len({v['speaker'] for v in ann.values()})} speakers. "
         "Counts come from acoustic detectors with limited recall: read them as lower bounds.", "",
         "| phrase | n | median span s | median voiced bursts | clips with a block ≥0.25 s | with a pause 0.12-0.25 s | with a repetition | with a prolongation | with a restart/repeat of the phrase |", "|---|---|---|---|---|---|---|---|---|"]
    for ph, rows in sorted(by.items()):
        n = len(rows); f = lambda c: f"{sum(1 for r in rows if c(r))}/{n}"
        L.append(f"| {ph} | {n} | {np.median([r['span_s'] or 0 for r in rows]):.2f} | {np.median([r['n_voiced'] for r in rows]):.0f} | "
                 f"{f(lambda r: r['blocks_s'])} | {f(lambda r: r['pauses_s'])} | {f(lambda r: r['repetitions'])} | {f(lambda r: r['prolongations_s'])} | {f(lambda r: r['restarts'])} |")
    # speaker consistency: does a speaker's number of voiced bursts in one phrase predict another phrase?
    sp = defaultdict(dict)
    for v in ann.values(): sp[v["speaker"]].setdefault(v["phrase"], []).append(v["n_voiced"])
    pairs = [(np.mean(d["hey_google"]), np.mean(d["hey_siri"])) for d in sp.values() if "hey_google" in d and "hey_siri" in d]
    if len(pairs) > 5:
        A = np.array(pairs); r = np.corrcoef(A[:, 0], A[:, 1])[0, 1]
        # permutation p-value
        pr = np.random.default_rng(0); perm = [np.corrcoef(A[:, 0], pr.permutation(A[:, 1]))[0, 1] for _ in range(2000)]
        L += ["", f"Speaker consistency (voiced bursts, hey_google vs hey_siri, n={len(A)} speakers): Pearson r = {r:.2f}, permutation p = {np.mean(np.abs(perm) >= abs(r)):.3f}."]
    # position of events: first third / middle / last third of the span
    pos = [0, 0, 0]
    for v in ann.values():
        if v["span_s"]:
            for b in v["blocks_s"] + v["pauses_s"]: pass
    (out / "pattern_report.md").write_text("\n".join(L) + "\n")
    # ---- synthetic dataset, split by speaker so that no voice is in two splits
    spk = sorted({v["speaker"] for v in ann.values()}); pr = np.random.default_rng(1).permutation(len(spk))
    split = {spk[i]: ("test" if j < len(spk) * 0.2 else "val" if j < len(spk) * 0.4 else "train") for j, i in enumerate(pr)}
    rows = []
    for sid, x in clips.items():
        v = ann[sid]
        for k in range(a.variants):
            ops = []
            y = x
            kinds = list(robust.DISFLUENCIES)
            n_ops = int(rng.integers(1, 4))
            for _ in range(n_ops):
                name = kinds[int(rng.integers(0, len(kinds)))]
                y = robust.DISFLUENCIES[name](y, rng); ops.append(name)
            y = y[: int(10 * SR)]
            fn = f"{sid}_v{k:02d}.wav"; save_wav(out / "synthetic" / fn, y)
            rows.append({"file": f"synthetic/{fn}", "source_clip": sid, "speaker": v["speaker"], "phrase": v["phrase"],
                         "is_hey_google": int(v["phrase"] == "hey_google"), "split": split[v["speaker"]], "operations": "+".join(ops),
                         "duration_s": round(len(y) / SR, 2)})
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(ann)} real clips preprocessed, {len(rows)} stuttered variants -> {out.resolve()}")


if __name__ == "__main__":
    main()

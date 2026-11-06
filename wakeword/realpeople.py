"""Recordings of REAL people who stutter  ->  clean clips, honest evaluation, speaker-independent fine-tuning.

Input: one folder of recordings named  <phrase>_spk<N>.<ext>   (ogg / wav / mp3 / m4a / opus ...), e.g.
    hey_google_spk3.ogg   alexa_spk3.ogg   hey_siri_spk3.ogg
Clips have different lengths, levels and noise; nothing about them is assumed.  The file name is the label.

    prepare   decode -> 16 kHz mono -> remove DC / rumble -> undo clipping -> fill digital silence -> trim -> level
              writes real_people/clean/*.wav and real_people/manifest.csv   (python run.py real-prepare SRC)
    evaluate  score a model on the clean clips: per clip, per speaker, AUC, threshold table (python run.py real-eval)
    cv        leave-speakers-out cross-validated fine-tuning: the only honest way to use ~14 speakers for BOTH
              training and testing (python run.py real-cv)
    final     fine-tune on all speakers and install the model (python run.py real-final)

Why speaker-level cross-validation: with one clip per speaker, any clip the model trained on is no longer a test of
generalisation.  Each speaker is therefore held out exactly once and scored by a model that never heard that voice.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from . import audio_io, dsp
from .config import BG, GLE, GOO, HEY, HOP, IGNORE, OTHER, ROOT, SR, WIN
from .features import frame_db

REAL_DIR = ROOT / "real_people"
AUDIO_EXT = (".ogg", ".opus", ".wav", ".mp3", ".m4a", ".aac", ".flac", ".mpeg", ".mp4", ".webm")
NAME_RE = re.compile(r"^(?P<phrase>.+?)_(?:spk|speaker|s)(?P<spk>\d+)(?:[_-](?P<take>\d+))?$", re.I)

# phrase -> (is the wake word, negative category used by the dataset / verifier weighting)
PHRASES = {
    "hey_google": (True, "real_wake"),
    "ok_google": (False, "near_miss"), "okay_google": (False, "near_miss"),
    "hey": (False, "partial"), "google": (False, "partial"),
    "alexa": (False, "other_wake"), "hey_alexa": (False, "other_wake"), "ok_alexa": (False, "other_wake"),
    "siri": (False, "other_wake"), "hey_siri": (False, "other_wake"),
    "computer": (False, "other_wake"), "hey_computer": (False, "other_wake"),
}
LEAD_S, TAIL_S = 0.40, 0.60            # silence kept before / after the speech when trimming
MANIFEST_FIELDS = ["sample_id", "speaker", "phrase", "is_wake_word", "category", "original", "clean",
                   "orig_seconds", "clean_seconds", "speech_seconds", "peak_before", "digital_silence_s", "notes"]


def parse_name(path):
    """('hey_google', speaker_id:int, take:int) or None."""
    m = NAME_RE.match(Path(path).stem)
    if not m:
        return None
    return m.group("phrase").lower(), int(m.group("spk")), int(m.group("take") or 1)


# ----------------------------------------------------------------------------- preprocessing
def speech_span(x: np.ndarray):
    """(first, last) voiced sample using the same energy rule as the rest of the project, or None."""
    db = frame_db(x)
    if db.size < 10:
        return None
    floor, peak = float(np.percentile(db, 10)), float(db.max())
    voiced = np.where(db > max(floor + 10.0, peak - 35.0))[0]
    if voiced.size == 0:
        return None
    return int(voiced[0]) * HOP, min(len(x), int(voiced[-1]) * HOP + WIN), int(voiced.size)


def fill_digital_silence(x: np.ndarray, rng: np.random.Generator, min_run: int = 400, level_db: float = -72.0,
                         gate_db: float = -82.0):
    """Phone / chat apps gate the noise to (almost) digital zero between words: after decoding, those stretches sit
    at about -94 dBFS (one 16-bit step) instead of at a room tone.  The detector's features assume a room tone, so
    stretches of >= 25 ms below `gate_db` get faint comfort noise (-72 dBFS, cross-faded).  Returns (y, seconds)."""
    db = frame_db(x)
    if db.size == 0:
        return x, 0.0
    z = np.zeros(len(x), dtype=bool)
    for i in np.where(db < gate_db)[0]:
        z[i * HOP: i * HOP + HOP] = True            # the frames' hop-sized core (frames overlap by 15 ms)
    if not z.any():
        return x, 0.0
    d = np.diff(np.concatenate([[0], z.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    y, filled = x.copy(), 0
    amp = 10.0 ** (level_db / 20.0)
    for s, e in zip(starts, ends):
        if e - s < min_run:
            continue
        n = e - s
        noise = (rng.standard_normal(n) * amp).astype(np.float32)
        f = min(80, n // 2)
        if f > 0:
            ramp = np.linspace(0, 1, f, dtype=np.float32)
            noise[:f] *= ramp
            noise[-f:] *= ramp[::-1]
        y[s:e] = noise
        filled += n
    return y, filled / SR


def clean(x: np.ndarray, seed: int = 0, fill_silence: bool = False):
    """Raw decoded clip -> clean clip.  Returns (y, info).  Nothing here changes WHAT was said or HOW it was
    stuttered: only level, clipping, DC / rumble and surplus silence (and, if asked, digital holes).

    fill_silence is OFF by default: on the 42 clips it did not improve the baseline model's separation (AUC 0.84 vs
    0.83, inside the noise of 14 positives), so it stays an option rather than an unverified assumption."""
    rng = np.random.default_rng(seed)
    info = {"peak_before": float(np.abs(x).max()), "notes": []}
    y = np.nan_to_num(x.astype(np.float32))
    y = y - float(y.mean())
    y = dsp.highpass(y, 50.0)
    pk = float(np.abs(y).max())
    if pk > 0.98:                                     # decoded opus overshoots full scale: scale down, never hard-clip
        y = y * (0.9 / pk)
        info["notes"].append("overshoot_scaled")
    dig = 0.0
    if fill_silence:
        y, dig = fill_digital_silence(y, rng)
        if dig > 0.2:
            info["notes"].append("digital_silence_filled")
    info["digital_silence_s"] = round(dig, 3)
    sp = speech_span(y)
    if sp is None:
        info.update(speech_seconds=0.0)
        info["notes"].append("NO_SPEECH_FOUND")
        return y, info
    a, b, nvoiced = sp
    lo, hi = max(0, a - int(LEAD_S * SR)), min(len(y), b + int(TAIL_S * SR))
    if lo > 0 or hi < len(y):
        info["notes"].append(f"trimmed_{lo / SR:.2f}s_head_{(len(y) - hi) / SR:.2f}s_tail")
    y = y[lo:hi]
    pk = float(np.abs(y).max())
    y = y * (0.8 / pk) if pk > 1e-6 else y            # peak 0.8: headroom, comparable level across clips
    info["speech_seconds"] = round(nvoiced / 100.0, 2)
    return y.astype(np.float32), info


def find_recordings(src):
    src = Path(src)
    return sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT)


def prepare(src, out_dir=REAL_DIR, log=print, fill_silence: bool = False):
    """Decode + clean every recording in `src`; copy the originals; write manifest.csv.  Returns the manifest rows."""
    out_dir = Path(out_dir)
    (out_dir / "original").mkdir(parents=True, exist_ok=True)
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for p in find_recordings(src):
        parsed = parse_name(p)
        if parsed is None or parsed[0] not in PHRASES:
            skipped.append(p.name)
            continue
        phrase, spk, take = parsed
        wake, cat = PHRASES[phrase]
        try:
            x = audio_io.load_audio(p)
        except Exception as exc:
            skipped.append(f"{p.name} (cannot decode: {exc})")
            continue
        y, info = clean(x, seed=spk * 100 + take, fill_silence=fill_silence)
        sid = f"{phrase}_spk{spk:02d}" + (f"_t{take}" if take > 1 else "")
        shutil.copy2(p, out_dir / "original" / p.name)
        audio_io.save_wav(out_dir / "clean" / f"{sid}.wav", y)
        rows.append({"sample_id": sid, "speaker": f"S{spk:02d}", "phrase": phrase, "is_wake_word": int(wake),
                     "category": cat, "original": f"original/{p.name}", "clean": f"clean/{sid}.wav",
                     "orig_seconds": round(len(x) / SR, 2), "clean_seconds": round(len(y) / SR, 2),
                     "speech_seconds": info.get("speech_seconds", 0.0), "peak_before": round(info["peak_before"], 3),
                     "digital_silence_s": info.get("digital_silence_s", 0.0), "notes": ";".join(info["notes"])})
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    spk = sorted({r["speaker"] for r in rows})
    npos = sum(r["is_wake_word"] for r in rows)
    log(f"prepared {len(rows)} clips from {len(spk)} speakers: {npos} 'Hey Google' + {len(rows) - npos} other phrases  -> {out_dir}")
    if skipped:
        log(f"  skipped {len(skipped)} file(s) (name not '<phrase>_spk<N>' or unknown phrase): " + ", ".join(skipped[:8]))
    bad = [r["sample_id"] for r in rows if "NO_SPEECH_FOUND" in r["notes"]]
    if bad:
        log("  WARNING no speech found in: " + ", ".join(bad))
    return rows


def load_manifest(out_dir=REAL_DIR):
    p = Path(out_dir) / "manifest.csv"
    if not p.exists():
        raise SystemExit(f"{p} not found - run:  python run.py real-prepare \"<folder with the recordings>\"")
    with open(p, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["is_wake_word"] = int(r["is_wake_word"])
    return rows


def load_clip(row, out_dir=REAL_DIR) -> np.ndarray:
    return audio_io.load_audio(Path(out_dir) / row["clean"])


# ----------------------------------------------------------------------------- evaluation
def _auc(y, s):
    try:
        from sklearn.metrics import roc_auc_score
        return round(float(roc_auc_score(y, s)), 4) if len(set(y)) == 2 else None
    except Exception:
        return None


def wilson(k: int, n: int, z: float = 1.96):
    """95 % Wilson interval of a proportion - with 14 positives the interval is wide, and the report must say so."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(float(max(0, c - h)), 3), round(float(min(1, c + h)), 3)


def score_clips(detector, rows, out_dir=REAL_DIR):
    """[{id, speaker, phrase, wake, score, hit, heard, reason}] for every clip."""
    recs = []
    for r in rows:
        res = detector.analyze(load_clip(r, out_dir), repair=False, with_curves=False)
        recs.append({"id": r["sample_id"], "speaker": r["speaker"], "phrase": r["phrase"], "wake": int(r["is_wake_word"]),
                     "category": r["category"], "score": float(res["score"]), "hit": bool(res["detected"]),
                     "heard": res.get("heard_text", ""), "reason": res.get("reason", "")})
    return recs


def summarise(recs, threshold, tag=""):
    pos = [r for r in recs if r["wake"]]
    neg = [r for r in recs if not r["wake"]]
    tp, fa = sum(r["hit"] for r in pos), sum(r["hit"] for r in neg)
    out = {"tag": tag, "threshold": round(float(threshold), 4), "n_wake": len(pos), "n_negative": len(neg),
           "detected": tp, "recall": round(tp / max(len(pos), 1), 4), "recall_95ci": wilson(tp, len(pos)),
           "false_accepts": fa, "FAR": round(fa / max(len(neg), 1), 4), "FAR_95ci": wilson(fa, len(neg)),
           "AUC": _auc([r["wake"] for r in recs], [r["score"] for r in recs]),
           "by_phrase": {}}
    for ph in sorted({r["phrase"] for r in recs}):
        sub = [r for r in recs if r["phrase"] == ph]
        out["by_phrase"][ph] = {"n": len(sub), "accepted": sum(r["hit"] for r in sub)}
    sw = []
    for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        sw.append({"threshold": t, "recall": round(float(np.mean([r["score"] >= t for r in pos])), 3) if pos else None,
                   "FAR": round(float(np.mean([r["score"] >= t for r in neg])), 3) if neg else None})
    out["threshold_sweep"] = sw
    return out


def evaluate(detector, out_dir=REAL_DIR, tag="current", write=True, log=print):
    rows = load_manifest(out_dir)
    recs = score_clips(detector, rows, out_dir)
    s = summarise(recs, detector.threshold, tag)
    s["clips"] = recs
    if write:
        (Path(out_dir) / f"eval_{tag}.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        (Path(out_dir) / f"eval_{tag}.md").write_text(markdown(s), encoding="utf-8")
    log(markdown(s).encode("ascii", "replace").decode("ascii"))
    return s


def markdown(s) -> str:
    L = [f"# Real people who stutter - {s['tag']}", "",
         f"Operating threshold {s['threshold']} (chosen on the project's validation data, NOT on these clips).", "",
         f"- 'Hey Google' accepted: **{s['detected']} / {s['n_wake']}**  (recall {s['recall']}, 95% CI {s['recall_95ci'][0]}-{s['recall_95ci'][1]})",
         f"- other phrases wrongly accepted: **{s['false_accepts']} / {s['n_negative']}**  (FAR {s['FAR']}, 95% CI {s['FAR_95ci'][0]}-{s['FAR_95ci'][1]})",
         f"- AUC (threshold-free separation of 'Hey Google' from the other phrases): **{s['AUC']}**", "",
         "| phrase | clips | accepted |", "|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['accepted']} |" for k, v in s["by_phrase"].items()]
    L += ["", "| threshold | recall | FAR |", "|---|---|---|"]
    L += [f"| {t['threshold']} | {t['recall']} | {t['FAR']} |" for t in s["threshold_sweep"]]
    if s.get("clips"):
        L += ["", "| clip | score | accepted | what the detector heard / why not |", "|---|---|---|---|"]
        for r in sorted(s["clips"], key=lambda r: (r["phrase"], r["speaker"])):
            L.append(f"| {r['id']} | {r['score']:.3f} | {'yes' if r['hit'] else 'no'} | {r['heard'] or r['reason']} |")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- frame labels of the real clips
SPEECH_PENALTY = -4.0          # per voiced frame explained as 'garbage' (outside the phrase): the file name says the clip IS the phrase


def _viterbi_emit(E: np.ndarray, g):
    """Viterbi over the decoder grammar with ready-made emissions E (T, N).  Same recursion as decoder.viterbi."""
    from .decoder import NEG
    T, N = E.shape
    delta = np.full((T, N), NEG, dtype=np.float32)
    psi = np.zeros((T, N), dtype=np.int16)
    delta[0] = g.start + E[0]
    for t in range(1, T):
        cand = delta[t - 1][:, None] + g.logA
        psi[t] = np.argmax(cand, axis=0)
        delta[t] = cand[psi[t], np.arange(N)] + E[t]
    final = delta[-1] + g.end
    s = int(np.argmax(final))
    if final[s] < NEG / 2:
        return None
    path = np.zeros(T, dtype=np.int16)
    path[-1] = s
    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path


def align_known_phrase(model, x: np.ndarray, min_unit_voiced: float = 0.4, max_bg_voiced: float = 0.25):
    """Frame labels (bg / hey / goo / gle) of a clip KNOWN to contain 'Hey Google' (said fluently or stuttered).

    The disfluency grammar of the decoder is aligned with Viterbi, with ONE addition over align.align_keyword_clip:
    speech frames may not be explained as 'garbage' before / after the phrase (SPEECH_PENALTY per frame).  The plain
    forced alignment leaves a clip's speech unlabelled whenever the current model does not yet hear the speaker's
    Hey / Goo / gle (14 of 14 clips here, 11 of them rejected); the model's posteriors then only decide WHERE the
    Hey|Goo|gle boundaries and repetitions lie, not WHETHER the speech belongs to the phrase.
    Returns (labels, info) or (None, reason)."""
    from . import decoder
    from .negatives import vad_labels
    lp = model.predict_wave(x)
    T = lp.shape[0]
    if T < 20:
        return None, "too short"
    g = decoder._GRAMMAR
    E = decoder._emissions(lp, g)
    voiced = vad_labels(x)[:T] != BG
    if voiced.size < T:
        voiced = np.pad(voiced, (0, T - voiced.size))
    garb = [i for i, c in enumerate(g.emit) if c == -1]
    for i in garb:
        E[voiced, i] += SPEECH_PENALTY
    path = _viterbi_emit(E, g)
    if path is None:
        return None, "no grammar path"
    kinds = np.array([g.names[s][0] for s in path])
    lab = np.full(T, BG, dtype=np.int64)
    lab[kinds == "H"], lab[kinds == "G"], lab[kinds == "L"] = HEY, GOO, GLE
    for k, mn, nm in (("H", 4, "Hey"), ("G", 4, "Goo"), ("L", 3, "gle")):
        if (kinds == k).sum() < mn:
            return None, f"{nm} too short ({int((kinds == k).sum())} frames)"
    for c, nm in ((HEY, "Hey"), (GOO, "Goo"), (GLE, "gle")):
        if voiced[lab == c].mean() < min_unit_voiced:
            return None, f"{nm} not on speech"
    nv = max(int(voiced.sum()), 1)
    if (voiced & (lab == BG)).sum() / nv > max_bg_voiced:
        return None, "too much speech labelled as pause"
    mean_p = {nm: float(np.exp(lp[lab == c, c]).mean()) for c, nm in ((HEY, "Hey"), (GOO, "Goo"), (GLE, "gle"))}
    return lab, {"mean_posterior": {k: round(v, 2) for k, v in mean_p.items()},
                 "frames": {nm: int((lab == c).sum()) for c, nm in ((HEY, "Hey"), (GOO, "Goo"), (GLE, "gle"))}}


def frame_labels_negative(x: np.ndarray, phrase: str) -> np.ndarray:
    """bg / other labels of a clip that is NOT the wake word.  Phrases that start with 'hey' / 'ok' contain a real
    'Hey' (and 'Google'-like sounds), so their speech frames are IGNOREd (as the dataset does for look-alike words);
    anything else ('alexa', 'siri', ...) is plain other speech."""
    from .negatives import vad_labels
    lab = vad_labels(x).copy()
    if phrase.startswith(("hey", "ok", "okay")) or phrase in ("google",):
        lab[lab == OTHER] = IGNORE
    return lab

"""Use the real-stuttering SEP-28k audio (and FluencyBank clips) that you already own.

What it does with a folder or .zip of 3-second clips (+ optional label CSV):
    scan       what is in there (clips, shows, episodes, label counts, durations)
    evaluate   false accepts per hour of REAL stuttered speech, per stuttering type  (a real-data result for the paper)
    calibrate  measure real gap / prolongation / block durations -> calibration.json used by the synthesiser
    pool       real stuttered clips as extra NEGATIVES for training (split by show + episode)

Clip names follow the SEP-28k convention  <Show>_<EpisodeId>_<ClipId>.wav  (any sub-folder layout works).
The label file is a CSV with the columns Show, EpId, ClipId, Prolongation, Block, SoundRep, WordRep, Interjection,
NoStutteredWords, PoorAudioQuality, Music, NoSpeech ... (annotator counts 0-3). Column names are matched case-insensitively
and missing columns are simply ignored.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import audio_io
from .config import HOP, ROOT, SR, WIN
from .segmenter import voiced_regions

AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3", ".m4a")
NAME_RE = re.compile(r"^(?P<show>.+?)_(?P<ep>\d+)_(?P<clip>\d+)$")
EVENTS = ["Prolongation", "Block", "SoundRep", "WordRep", "Interjection", "NoStutteredWords"]
CALIBRATION_FILE = ROOT / "calibration.json"


class Clip:
    __slots__ = ("key", "show", "ep", "clip", "path", "zip_path", "member")

    def __init__(self, key, show, ep, clip, path=None, zip_path=None, member=None):
        self.key, self.show, self.ep, self.clip = key, show, ep, clip
        self.path, self.zip_path, self.member = path, zip_path, member

    @property
    def group(self):
        return f"{self.show}_{self.ep}"


class Store:
    """Folder(s) and/or zip file(s) holding clips; zips are opened lazily and kept open."""

    def __init__(self, root):
        self.root = Path(root)
        self.clips: list[Clip] = []
        self.label_files: list = []          # (display name, reader callable)
        self._zips: dict = {}
        self._index()

    def _add(self, name, **kw):
        stem = Path(name).stem
        m = NAME_RE.match(stem)
        if not m:
            # not a SEP-28k / FluencyBank clip (e.g. LibriSpeech, MUSAN or a WhatsApp recording in the same folder):
            # skip it.  v1.1 kept such files under a show called 'unknown', which mixed other corpora into the pool
            # and broke the show-level dev / test separation.
            self.skipped = getattr(self, "skipped", 0) + 1
            return
        show, ep, clip = m.group("show"), m.group("ep"), m.group("clip")
        self.clips.append(Clip(f"{show}_{ep}_{clip}", show, ep, clip, **kw))

    def _scan_zip(self, zp: Path):
        zf = zipfile.ZipFile(zp)
        self._zips[str(zp)] = zf
        for info in zf.infolist():
            low = info.filename.lower()
            if info.is_dir() or "__macosx" in low:
                continue
            if low.endswith(AUDIO_EXT):
                self._add(info.filename, zip_path=str(zp), member=info.filename)
            elif low.endswith(".csv") and "label" in low:
                self.label_files.append((f"{zp.name}:{info.filename}", lambda zf=zf, n=info.filename: zf.read(n).decode("utf-8", "replace")))
            elif low.endswith(".zip"):                       # zip inside zip: one level, in memory
                try:
                    inner = zipfile.ZipFile(io.BytesIO(zf.read(info.filename)))
                except Exception:
                    continue
                for ii in inner.infolist():
                    il = ii.filename.lower()
                    if not ii.is_dir() and il.endswith(AUDIO_EXT):
                        self._add(ii.filename, zip_path=f"{zp}!{info.filename}", member=ii.filename)
                        self._zips[f"{zp}!{info.filename}"] = inner

    def _index(self):
        if self.root.is_file() and self.root.suffix.lower() == ".zip":
            self._scan_zip(self.root)
        else:
            for p in sorted(self.root.rglob("*")):
                low = p.name.lower()
                if p.is_dir():
                    continue
                if low.endswith(AUDIO_EXT):
                    self._add(p.name, path=str(p))
                elif low.endswith(".csv") and "label" in low:
                    self.label_files.append((str(p), lambda p=p: p.read_text(encoding="utf-8", errors="replace")))
                elif low.endswith(".zip"):
                    self._scan_zip(p)

    def load(self, c: Clip) -> np.ndarray:
        if c.path:
            return audio_io.load_audio(c.path)
        zf = self._zips[c.zip_path]
        return audio_io.load_audio_bytes(zf.read(c.member))

    def labels(self) -> dict:
        """{clip key: {event: annotator count}} merged from every label CSV found."""
        out = {}
        for name, reader in self.label_files:
            try:
                rows = list(csv.DictReader(io.StringIO(reader())))
            except Exception:
                continue
            if not rows:
                continue
            cols = {c.lower().strip(): c for c in rows[0].keys()}
            if not {"show", "epid", "clipid"} <= set(cols):
                continue
            for r in rows:
                key = f"{r[cols['show']].strip()}_{r[cols['epid']].strip()}_{r[cols['clipid']].strip()}"
                d = {}
                for ev in EVENTS + ["Unsure", "PoorAudioQuality", "Music", "NoSpeech", "NaturalPause", "DifficultToUnderstand"]:
                    c = cols.get(ev.lower())
                    if c:
                        try:
                            d[ev] = int(float(r[c]))
                        except (ValueError, TypeError):
                            d[ev] = 0
                out[key] = d
        return out


def labelled(clips, labs: dict):
    """Only clips that appear in a SEP-28k / FluencyBank label CSV (when label files exist)."""
    return [c for c in clips if c.key in labs] if labs else clips


def usable(lab: dict | None) -> bool:
    """Speech clip of acceptable quality (unlabelled clips are kept)."""
    if not lab:
        return True
    return lab.get("PoorAudioQuality", 0) < 2 and lab.get("Music", 0) < 2 and lab.get("NoSpeech", 0) < 2


def dominant_event(lab: dict | None) -> str:
    if not lab:
        return "unlabelled"
    ev = [e for e in EVENTS[:5] if lab.get(e, 0) >= 2]
    if len(ev) == 1:
        return ev[0]
    if len(ev) > 1:
        return "multiple"
    return "NoStutteredWords" if lab.get("NoStutteredWords", 0) >= 2 else "unsure"


# ----------------------------------------------------------------------------- show-level parts
def split_shows(clips, shares=(("train", 0.70), ("dev", 0.15), ("test", 0.15)), min_shows: int = 6) -> dict:
    """{show: part}.  Whole shows (podcasts) go to ONE part, so no host or guest voice is shared between the
    training pool, the dev pool (threshold tuning) and the test pool (final report).  Shows are handed out
    largest first to the part with the biggest shortfall, so the clip shares stay close to the targets even
    though a few shows are huge.  With fewer than `min_shows` shows a show-level split is impossible, so every
    show then belongs to every part ("all")."""
    cnt = Counter(c.show for c in clips)
    if len(cnt) < min_shows:
        return {s: "all" for s in cnt}
    total, got, out = sum(cnt.values()), {p: 0 for p, _ in shares}, {}
    for show, n in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
        p = max(shares, key=lambda ps: ps[1] * total - got[ps[0]])[0]
        out[show] = p
        got[p] += n
    return out


def in_part(parts: dict, c, part: str) -> bool:
    return part == "all" or parts.get(c.show) in (part, "all")


def neutral_group(show: str) -> str:
    """Stable, anonymous group id of a show (no source names in the dataset files)."""
    return "RS_" + hashlib.md5(show.encode("utf-8")).hexdigest()[:6]


MIN_ALLOWED_FA = 10          # a rate is only trusted when the audio is long enough to allow at least this many false accepts


def threshold_for_rate(scores_desc, hours: float, per_hour: float, min_allowed: int = MIN_ALLOWED_FA):
    """Lowest threshold that lets at most `per_hour` false accepts per hour through on these clips, or None when the
    audio is too short to say (it must allow at least `min_allowed` false accepts, otherwise the threshold would be
    set by one or two clips).  Scores are the 4-decimal scores the detector reports, hence the +1e-4 step."""
    k = int(np.floor(per_hour * hours))                  # false accepts we may allow
    if k < min_allowed:
        return None
    if k >= len(scores_desc):
        return 0.0
    return float(min(1.0, round(float(scores_desc[k]) + 1e-4, 4)))


# ----------------------------------------------------------------------------- scan
def scan(root, sample: int = 200, log=print) -> dict:
    st = Store(root)
    labs = st.labels()
    n = len(st.clips)
    rng = np.random.default_rng(0)
    durs = []
    for i in rng.permutation(n)[:sample]:
        try:
            durs.append(len(st.load(st.clips[int(i)])) / SR)
        except Exception:
            pass
    shows = Counter(c.show for c in st.clips)
    groups = {c.group for c in st.clips}
    kinds = Counter(dominant_event(labs.get(c.key)) for c in st.clips)
    matched = sum(1 for c in st.clips if c.key in labs)
    res = {"root": str(root), "n_clips": n, "n_shows": len(shows), "n_episodes": len(groups), "shows": dict(shows.most_common(15)),
           "label_files": [n_ for n_, _ in st.label_files], "n_label_rows": len(labs), "clips_with_labels": matched,
           "dominant_event_counts": dict(kinds), "usable_clips": sum(1 for c in st.clips if usable(labs.get(c.key))),
           "mean_clip_seconds": round(float(np.mean(durs)), 2) if durs else None,
           "est_hours": round(n * (float(np.mean(durs)) if durs else 3.0) / 3600, 1)}
    log(json.dumps(res, indent=2))
    return res


# ----------------------------------------------------------------------------- evaluation on real stuttering
EVAL_TARGETS = (5, 2, 1, 0.5)                     # false accepts per hour that the report translates into thresholds
EVAL_GRID = (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999)


def evaluate(detector, root, max_clips: int = 5000, seed: int = 0, out_dir=None, log=print, part: str = "test",
             targets=EVAL_TARGETS) -> dict:
    """False accepts of the wake-word detector on REAL stuttered speech (none of it is 'Hey Google').

    `part` picks the shows to use: "test" (final report), "dev" (threshold tuning), "train" (what training may use)
    or "all".  Parts are whole shows, so a voice never sits in two of them."""
    st = Store(root)
    labs = st.labels()
    parts = split_shows(labelled(st.clips, labs))
    clips = [c for c in labelled(st.clips, labs) if usable(labs.get(c.key)) and in_part(parts, c, part)]
    if not clips:
        raise SystemExit(f"no usable clips in part '{part}' of {root}")
    by_part = defaultdict(list)
    for show, pt in sorted(parts.items()):
        by_part[pt].append(show)
    log(f"      show split ({len(parts)} shows, {getattr(st, 'skipped', 0)} non-SEP-28k files ignored): "
        + "   ".join(f"{pt}={','.join(v)}" for pt, v in sorted(by_part.items())))
    if part in ("dev", "test") and "all" in by_part:
        log("      WARNING: too few shows for a show-level split - dev and test are the SAME audio; "
            "a threshold tuned on dev is not an unbiased test result.")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(clips))[:max_clips]
    total_s, fa, by_ev, hits, scores, shows = 0.0, 0, defaultdict(lambda: [0, 0.0, 0]), [], [], set()
    for k, i in enumerate(idx):
        c = clips[int(i)]
        try:
            x = st.load(c)
        except Exception:
            continue
        res = detector.analyze(x, repair=False, with_curves=False)
        dur = len(x) / SR
        ev = dominant_event(labs.get(c.key))
        total_s += dur
        scores.append(float(res.get("score", 0.0)))
        shows.add(c.show)
        e = by_ev[ev]
        e[0] += 1
        e[1] += dur
        if res["detected"]:
            fa += 1
            e[2] += 1
            hits.append({"clip": c.key, "score": res["score"], "event": ev})
        if (k + 1) % 500 == 0:
            log(f"      {k + 1}/{len(idx)} clips, {fa} false accepts so far")
    hours = total_s / 3600
    out = {"clips_evaluated": int(sum(v[0] for v in by_ev.values())), "hours": round(hours, 3), "false_accepts": fa,
           "false_accepts_per_hour": round(fa / max(hours, 1e-9), 2),
           "by_event": {k: {"clips": v[0], "hours": round(v[1] / 3600, 3), "false_accepts": v[2],
                            "per_hour": round(v[2] / max(v[1] / 3600, 1e-9), 2)} for k, v in sorted(by_ev.items())},
           "false_accept_clips": sorted(hits, key=lambda h: -h["score"])[:50], "threshold": detector.threshold,
           "part": part, "n_shows": len(shows), "shows": sorted(shows)}
    sc = np.sort(np.asarray(scores))[::-1]
    out["false_accepts_per_hour_by_threshold"] = {str(t): round(float((sc >= t).sum()) / max(hours, 1e-9), 2) for t in EVAL_GRID}
    out["threshold_for_fa_per_hour"] = {str(t): threshold_for_rate(sc, hours, t) for t in targets}
    out["resolution_fa_per_hour"] = round(1.0 / max(hours, 1e-9), 3)     # one false accept = this many per hour
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "sep28k_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"real stuttered speech [{part} shows: {len(shows)}]: {out['clips_evaluated']} clips, {out['hours']} h -> {fa} false accepts "
        f"({out['false_accepts_per_hour']} per hour)")
    return out


def tune_threshold(detector, root, model_dir, per_hour: float, max_clips: int = 5000, seed: int = 0, log=print) -> dict:
    """Set the decision threshold from the DEV shows: the lowest threshold that allows at most `per_hour` false
    accepts per hour of real speech.  It is written into verifier.json (the first original is kept as verifier.json.bak).
    Report the result afterwards on the TEST shows (python run.py sep28k-eval PATH) and on the synthetic split."""
    vp = Path(model_dir) / "verifier.json"
    if not vp.exists():
        raise SystemExit(f"{vp} not found: train a model first (python run.py all)")
    res = evaluate(detector, root, max_clips=max_clips, seed=seed, log=log, part="dev", targets=(per_hour,))
    new = res["threshold_for_fa_per_hour"][str(per_hour)]
    old = float(detector.threshold)
    if new is None:
        need = MIN_ALLOWED_FA / per_hour
        raise SystemExit(f"NOT tuned: {res['hours']} h of dev audio is too little to set {per_hour} false accepts/hour "
                         f"(need about {need:.0f} h, i.e. use --max-clips with a larger number, or a higher target). "
                         f"The threshold stays {old:.4f}.")
    d = json.loads(vp.read_text(encoding="utf-8"))
    bak = vp.with_name(vp.name + ".bak")
    if not bak.exists():
        bak.write_text(json.dumps(d), encoding="utf-8")
    d["threshold"] = new
    vp.write_text(json.dumps(d), encoding="utf-8")
    log(f"threshold {old:.4f} -> {new:.4f}  (target {per_hour} false accepts/hour on {res['hours']} h of dev speech from {res['n_shows']} shows)")
    return {"old_threshold": old, "new_threshold": new, "per_hour": per_hour, "dev_hours": res["hours"]}


# ----------------------------------------------------------------------------- calibration of the synthesiser
def _runs(x):
    regs, _ = voiced_regions(x, min_frames=3, merge_gap=0.04)
    return regs


def calibrate(root, max_clips: int = 6000, seed: int = 0, out_file=CALIBRATION_FILE, log=print) -> dict:
    """Measure real durations from clips whose label says what happened (clip-level labels, annotator count >= 2)."""
    st = Store(root)
    labs = st.labels()
    if not labs:
        raise SystemExit("No label CSV found next to the clips (a file whose name contains 'label'). Calibration needs it.")
    rng = np.random.default_rng(seed)
    clips = [c for c in st.clips if c.key in labs and usable(labs[c.key])]
    gaps = defaultdict(list)           # event -> gaps between voiced runs (ms)
    longest_run = defaultdict(list)    # event -> longest voiced run (ms)
    for i in rng.permutation(len(clips))[:max_clips]:
        c = clips[int(i)]
        ev = dominant_event(labs[c.key])
        if ev not in ("Prolongation", "Block", "SoundRep", "WordRep", "NoStutteredWords"):
            continue
        try:
            x = st.load(c)
        except Exception:
            continue
        regs = _runs(x)
        if not regs:
            continue
        longest_run[ev].append(max(b - a for a, b in regs) * 10)
        for (a0, b0), (a1, b1) in zip(regs[:-1], regs[1:]):
            g = (a1 - b0) * 10
            if 30 <= g <= 3000:
                gaps[ev].append(g)
    pct = lambda v, p: float(np.percentile(v, p)) if len(v) >= 20 else None
    cal = {
        "source": str(root), "clips_used": {k: len(v) for k, v in longest_run.items()},
        "repetition_gap_ms": {"median": pct(gaps["SoundRep"] + gaps["WordRep"], 50), "p10": pct(gaps["SoundRep"] + gaps["WordRep"], 10),
                              "p90": pct(gaps["SoundRep"] + gaps["WordRep"], 90)},
        "fluent_gap_ms": {"median": pct(gaps["NoStutteredWords"], 50), "p90": pct(gaps["NoStutteredWords"], 90)},
        "block_gap_ms": {"p10": pct(gaps["Block"], 10), "median": pct(gaps["Block"], 50), "p90": pct(gaps["Block"], 90)},
        "prolongation_run_ms": {"p10": pct(longest_run["Prolongation"], 10), "median": pct(longest_run["Prolongation"], 50),
                                "p90": pct(longest_run["Prolongation"], 90)},
        "fluent_run_ms": {"median": pct(longest_run["NoStutteredWords"], 50), "p90": pct(longest_run["NoStutteredWords"], 90)},
        "note": "clip-level labels: gaps/runs are measured with an energy VAD, so they are indicative priors, not ground truth."}
    Path(out_file).write_text(json.dumps(cal, indent=2), encoding="utf-8")
    log(json.dumps(cal, indent=2))
    log(f"written {out_file}  (the synthesiser reads it automatically at the next  python run.py build-data)")
    return cal


# ----------------------------------------------------------------------------- training negatives
def as_float(x: np.ndarray) -> np.ndarray:
    """Pool clips are kept as int16 (half the memory of float32; lossless for 16-bit wav sources)."""
    return x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)


class _LazyClip:
    """Reads one pool clip on demand: int16 audio, at most 6 s, or None when it cannot be read or is too short."""
    __slots__ = ("store", "clip")

    def __init__(self, store, clip):
        self.store, self.clip = store, clip

    def __call__(self):
        try:
            x = self.store.load(self.clip)
        except Exception:
            return None
        if len(x) <= int(0.5 * SR):
            return None
        return np.clip(np.round(x[: int(6 * SR)] * 32768.0), -32768, 32767).astype(np.int16)


def load_pool(root, n: int, seed: int = 0, log=print, part: str = "train"):
    """[(group, 'real_stutter_sep28k', audio)] - real stuttered speech as extra negatives (quality-filtered).

    Only clips of the TRAIN shows are used (see split_shows), so the dev / test shows stay unseen.  The group is
    the show (anonymised), so the dataset's own train/val/test split never puts one voice into two splits.
    Clips without a label row are kept: they are only ever used as NEGATIVES (they are never wake-word examples)."""
    st = Store(root)
    labs = st.labels()
    parts = split_shows(labelled(st.clips, labs))
    clips = [c for c in labelled(st.clips, labs) if usable(labs.get(c.key)) and in_part(parts, c, part)]
    rng = np.random.default_rng(seed)
    # Nothing is read here: tens of thousands of clips would not fit in memory.  Each item carries a loader that
    # reads (and trims) its clip only when the dataset builder actually uses it.
    out = [(neutral_group(clips[int(i)].show), "real_stutter_sep28k", _LazyClip(st, clips[int(i)]))
           for i in rng.permutation(len(clips))[:n]]
    log(f"      real-speech negatives: {len(out)} clips from {len({g for g, _, _ in out})} shows")
    return out


def load_calibration() -> dict:
    try:
        return json.loads(Path(CALIBRATION_FILE).read_text(encoding="utf-8"))
    except Exception:
        return {}

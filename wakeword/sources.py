"""Loading the raw material the dataset is made from (all optional, all detected automatically).

Expected files in ./raw (names are matched by keyword, so renamed copies work too):
    original_heyGoogle_stutter_dataset*.zip   curated fluent 'Hey Google' clips (WhatsApp)
    OriginalRecordingsFluentHeyGoogle*.zip    raw WhatsApp recordings (old + newly added)
    synthetic_hey_google_dataset*.zip         neural-TTS fluent clips (extra voices)
    Dysarthria_and_Non_Dysarthria*.zip        TORGO-style speech  -> negative speech
    clips*.zip  (or a clips/ folder)          SEP-28k / FluencyBank real stuttering clips -> negatives
"""
from __future__ import annotations

import csv
import io
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio_io
from .config import RAW_DIR, SR
from .segmenter import Utterance, build_utterance


@dataclass
class Source:
    id: str
    group: str            # leakage-free grouping key (splits are done on this)
    kind: str             # real | tts
    utt: Utterance
    origin: str = ""
    flagged: bool = False


def find_zip(raw_dir: Path, key: str):
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return None
    for p in sorted(raw_dir.glob("*.zip")):
        if key.lower() in p.name.lower():
            return p
    return None


def _unit_sane(u: Utterance) -> bool:
    d = lambda a, b: (b - a) / SR
    return (0.06 <= d(u.start, u.b1) <= 0.7 and 0.06 <= d(u.b1, u.b2) <= 0.7 and 0.05 <= d(u.b2, u.end) <= 0.8)


def speech_qa(u: Utterance):
    """Do the Hey / Goo / gle labels sit on the speech?  (v1.2c)  Returns (ok, reason).

    v1.2b audit of the data: 38 of 100 real recordings had units placed on silence (e.g. 'Hey' + 'Goo' labelled
    0.6 s before the speaker started) or real speech labelled as background, and every one of the 60 clips made
    from such a recording inherited the error.  _unit_sane only checks durations; this checks the audio:
      - each unit is mostly voiced (Goo may contain the /g/ closure),
      - the 0.3 s just before 'Hey' and just after 'gle' are not speech (the keyword was not cut short),
      - the recording's 'noise' (used to fill synthetic pauses) contains no speech."""
    from .features import frame_db
    from .config import HOP
    db = frame_db(u.audio)
    if db.size < 10:
        return False, "too short"
    floor, peak = float(np.percentile(db, 10)), float(db.max())
    voiced = db > max(floor + 10.0, peak - 35.0)
    f = lambda smp: int(np.clip(smp // HOP, 0, len(db)))
    a, b1, b2, e = f(u.start), f(u.b1), f(u.b2), f(u.end)
    for name, lo, hi, need in (("Hey", a, b1, 0.5), ("Goo", b1, b2, 0.35), ("gle", b2, e, 0.5)):
        if hi <= lo or voiced[lo:hi].mean() < need:
            return False, f"{name} not on speech"
    w = 30
    if voiced[max(0, a - w):a].mean() > 0.5 if a > 0 else False:
        return False, "speech right before Hey (keyword cut short)"
    if voiced[e:e + w].mean() > 0.5 if e < len(voiced) else False:
        return False, "speech right after gle (keyword cut short)"
    if u.noise is not None and len(u.noise) > 800:
        nd = frame_db(u.noise)
        if nd.size and float(np.percentile(nd, 95)) > peak - 15.0:
            return False, "pause audio contains speech"
    return True, "ok"


def _read_manifest(zf: zipfile.ZipFile):
    with zf.open("manifest.csv") as fh:
        return list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))


def load_curated(raw_dir: Path, limit: int | None = None):
    """Unique curated fluent clips (one per original recording). Returns (sources, referenced_filenames)."""
    zp = find_zip(raw_dir, "original_heygoogle")
    if zp is None:
        return [], set()
    out, names, seen = [], set(), set()
    with zipfile.ZipFile(zp) as zf:
        rows = _read_manifest(zf)
        for r in rows:
            m = re.search(r"\((WhatsApp Audio[^)]*(?:\([0-9]+\))?[^)]*\.[A-Za-z0-9]+)\)", r.get("source_recording_id", ""))
            if m:
                names.add(m.group(1))
            if r["label"] != "fluent":
                continue
            key = r.get("source_recording_id", r["sample_id"])[:8]
            if key in seen:
                continue
            seen.add(key)
            x = audio_io.load_audio_bytes(zf.read(r["relative_path"]))
            u = build_utterance(x, meta={"origin": r["sample_id"]})
            if u is None:
                continue
            out.append(Source(id=f"REC_{key[-3:]}", group=f"REC_{key[-3:]}", kind="real", utt=u,
                              origin=r["relative_path"], flagged=not _unit_sane(u)))
            if limit and len(out) >= limit:
                break
    return out, names


def load_tts(raw_dir: Path, limit: int | None = None):
    zp = find_zip(raw_dir, "synthetic_hey_google")
    if zp is None:
        return []
    out = []
    with zipfile.ZipFile(zp) as zf:
        for r in _read_manifest(zf):
            if r["label"] != "fluent":
                continue
            x = audio_io.load_audio_bytes(zf.read(r["relative_path"]))
            u = build_utterance(x, meta={"origin": r["sample_id"]})
            if u is None:
                continue
            voice = r.get("voice_id", "tts")
            out.append(Source(id=f"TTS_{r['sample_id'][-4:]}_{voice}", group=f"TTS_{voice}", kind="tts", utt=u,
                              origin=r["relative_path"], flagged=not _unit_sane(u)))
            if limit and len(out) >= limit:
                break
    return out


def load_raw_recordings(raw_dir: Path, skip_names: set):
    """Decode raw WhatsApp recordings that are NOT already represented by a curated clip."""
    zp = find_zip(raw_dir, "originalrecordings")
    if zp is None:
        return []
    items = []
    with zipfile.ZipFile(zp) as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if info.is_dir():
                continue
            base = info.filename.split("/")[-1]
            if base in skip_names or base.startswith("."):
                continue
            try:
                x = audio_io.load_audio_bytes(zf.read(info.filename))
            except audio_io.AudioDecodeError as exc:
                print(f"  ! cannot decode {base}: {exc}")
                continue
            if len(x) > int(0.4 * SR):
                items.append((base, x))
    return items


# ----------------------------------------------------------------------------- external negatives
def load_torgo_pool(raw_dir: Path, n_per_speaker: int, rng: np.random.Generator):
    """Random speech clips from the dysarthria / control corpus -> list of (group, kind, audio)."""
    zp = find_zip(raw_dir, "dysarthria")
    if zp is None:
        return []
    out = []
    with zipfile.ZipFile(zp) as zf:
        by_spk: dict = {}
        for i in zf.infolist():
            if i.is_dir() or not i.filename.lower().endswith(".wav") or "/Wav/" not in i.filename:
                continue
            if i.file_size < 12000 or i.file_size > 230000:        # 0.4 s .. 7 s
                continue
            parts = i.filename.split("Dataset/")[-1].split("/")
            by_spk.setdefault((parts[0], parts[1]), []).append(i)
        for (cat, spk), infos in sorted(by_spk.items()):
            kind = "dysarthric" if "dysarthria" in cat.lower() and "non" not in cat.lower() else "control"
            idx = rng.permutation(len(infos))[: int(n_per_speaker * 1.4)]
            got = 0
            for k in idx:
                try:
                    x = audio_io.load_audio_bytes(zf.read(infos[k].filename))
                except Exception:
                    continue
                if np.abs(x).max() < 0.01:
                    continue
                out.append((spk, kind, x[: int(5.5 * SR)]))
                got += 1
                if got >= n_per_speaker:
                    break
    return out


def load_clips_pool(raw_dir: Path):
    """SEP-28k / FluencyBank 3 s clips (real people who stutter) -> (group, 'sep28k', audio)."""
    out = []
    folder = Path(raw_dir) / "clips"
    files = []
    zp = find_zip(raw_dir, "clips")
    if zp is not None:
        with zipfile.ZipFile(zp) as zf:
            for i in sorted(zf.infolist(), key=lambda i: i.filename):
                if i.filename.lower().endswith(".wav") and not i.is_dir():
                    files.append((i.filename.split("/")[-1], zf.read(i.filename)))
    elif folder.exists():
        for p in sorted(folder.rglob("*.wav")):
            files.append((p.name, p.read_bytes()))
    for name, buf in files:
        stem = name[:-4]
        m = re.match(r"(.+?)_(\d+)_(\d+)$", stem)
        group = f"{m.group(1)}_{m.group(2)}" if m else stem
        try:
            out.append((group, "sep28k", audio_io.load_audio_bytes(buf)))
        except Exception:
            pass
    return out


# ----------------------------------------------------------------------------- Piper-TTS corpora (multi-speaker, emotional)
_AMBIGUOUS = re.compile(r"^\s*(hey|hi|ok|okay),?\s+google", re.I)


def _find_member(zf: zipfile.ZipFile, suffix: str):
    for n in zf.namelist():
        if n.endswith(suffix):
            return n
    return None


def _open_corpus(raw_dir: Path, folder: str):
    """hey_google_ds / neg_ds can come as their own zip or inside hey_google_full.zip."""
    for key in (folder, "hey_google_full"):
        zp = find_zip(raw_dir, key)
        if zp is not None:
            zf = zipfile.ZipFile(zp)
            meta = _find_member(zf, f"{folder}/metadata.csv")
            if meta:
                return zf, meta[: -len("metadata.csv")]
            zf.close()
    return None, None


def load_piper_positives(raw_dir: Path, n_max: int, rng: np.random.Generator):
    """Fluent 'Hey Google' clips from the Piper corpus (>1000 synthetic speakers, 10 emotions, simulated ages).

    'Ok / Okay Google' rows are skipped: the wake word of this project is exactly 'Hey Google'.
    """
    zf, base = _open_corpus(raw_dir, "hey_google_ds")
    if zf is None:
        return []
    with zf:
        meta = list(csv.DictReader(io.TextIOWrapper(zf.open(base + "metadata.csv"), encoding="utf-8")))
        rows = [r for r in meta if r["text"].lower().replace(",", "").startswith("hey google")]
        # stratify over emotion x age so every condition is represented
        buckets: dict = {}
        for r in rows:
            buckets.setdefault((r["emotion"], r["age_group"]), []).append(r)
        order = []
        for k in sorted(buckets):
            order += [buckets[k][i] for i in rng.permutation(len(buckets[k]))]
        # interleave buckets
        picked, per = [], {k: [buckets[k][i] for i in rng.permutation(len(buckets[k]))] for k in buckets}
        while len(picked) < min(n_max, len(rows)):
            progressed = False
            for k in sorted(per):
                if per[k]:
                    picked.append(per[k].pop())
                    progressed = True
                    if len(picked) >= n_max:
                        break
            if not progressed:
                break
        out = []
        for r in picked:
            try:
                x = audio_io.load_audio_bytes(zf.read(base + r["file"]))
            except Exception:
                continue
            u = build_utterance(x, meta={"origin": r["file"], "emotion": r["emotion"], "age": r["age_group"]})
            if u is None:
                continue
            grp = f"PIPER_{r['voice']}:{r.get('speaker_id', '')}"
            out.append(Source(id="PIP_" + r["file"].split("/")[-1][:6], group=grp, kind="tts", utt=u,
                              origin=f"{r['voice']}|{r['emotion']}|{r['age_group']}", flagged=not _unit_sane(u)))
    return out


def load_piper_negatives(raw_dir: Path, per_category: dict, rng: np.random.Generator):
    """Labelled negatives (near-misses like 'Hey Poodle', 'Just google it later', other wake words, noise ...).

    Returns [(group, category, audio, text)]. Rows that actually start with 'Hey/OK Google' are dropped
    (e.g. 'Hey Google Maps') because they would contradict the wake-word label.
    """
    zf, base = _open_corpus(raw_dir, "neg_ds")
    if zf is None:
        return []
    out = []
    with zf:
        meta = list(csv.DictReader(io.TextIOWrapper(zf.open(base + "metadata.csv"), encoding="utf-8")))
        by_cat: dict = {}
        for r in meta:
            if _AMBIGUOUS.match(r["text"] or ""):
                continue
            by_cat.setdefault(r["category"], []).append(r)
        for cat, n in per_category.items():
            rows = by_cat.get(cat, [])
            for i in rng.permutation(len(rows))[:n]:
                r = rows[int(i)]
                try:
                    x = audio_io.load_audio_bytes(zf.read(base + r["file"]))
                except Exception:
                    continue
                if len(x) < int(0.2 * SR):
                    continue
                out.append((f"PIPERNEG_{r['voice']}:{r.get('speaker_id', '')}", cat, x[: int(6 * SR)], r["text"]))
    return out


# ----------------------------------------------------------------------------- recorded-style disfluent clips (v1.2)
def load_team_disfluent(raw_dir: Path, log=print):
    """The stuttered 'Hey Google' clips of the two project datasets (original_heyGoogle_stutter_dataset: spliced from
    the 42 real recordings; synthetic_hey_google_dataset: spliced from 6 neural voices).  v1.1 used only their FLUENT
    rows as sources, so the model never saw this (different) way of producing stuttered speech.

    Returns [dict(group, kind, x, dtype, word, sample_id)].  The group equals the group of the fluent source it was
    made from (REC_xxx / TTS_<voice>), so a clip always lands in the same split as its fluent original."""
    out = []
    for key, kind in (("original_heygoogle", "real"), ("synthetic_hey_google", "tts")):
        zp = find_zip(raw_dir, key)
        if zp is None:
            continue
        n = 0
        with zipfile.ZipFile(zp) as zf:
            for r in _read_manifest(zf):
                if r.get("label") != "disfluent":
                    continue
                if kind == "real":
                    k = r.get("source_recording_id", "")[:8]
                    if not k:
                        continue
                    group = f"REC_{k[-3:]}"
                else:
                    group = f"TTS_{r.get('voice_id', 'tts')}"
                try:
                    x = audio_io.load_audio_bytes(zf.read(r["relative_path"]))
                except Exception:
                    continue
                out.append({"group": group, "kind": kind, "x": x, "dtype": r.get("disfluency_type", "mixed"),
                            "word": r.get("affected_word", "none"), "sample_id": r.get("sample_id", "")})
                n += 1
        log(f"      recorded-style stuttered clips from {zp.name}: {n}")
    return out

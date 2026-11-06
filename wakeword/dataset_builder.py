"""Builds the complete dataset: fluent + synthetic-disfluent positives and negatives.

Output layout (data/):
    audio/fluent/*.wav        fluent 'Hey Google'        (wake word, label=fluent)
    audio/disfluent/*.wav     stuttered 'Hey Google'     (wake word, label=disfluent)
    audio/negative/*.wav      everything else            (NOT the wake word)
    manifest.csv              one row per clip, with split / group / type / timing metadata
    annotations.jsonl         frame-level segments + disfluency event timeline of each clip
    summary.json              counts per split / type
    source_audit.csv          which source recordings were used / rejected and why
"""
from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import audio_io, augment, negatives, sep28k, sources
from .config import DATA_DIR, MODEL_DIR, RAW_DIR, SR
from .disfluency import DISFLUENCY_TYPES, PROFILE_NAMES, PROFILE_WEIGHTS, DisfluencySynthesizer, Piece, _flatten, random_spec
from .config import IGNORE, OTHER
from .negatives import HARD_KINDS, NOISE_KINDS, external_pieces, labels_to_segments, vad_labels
from .segmenter import build_utterance

MANIFEST_FIELDS = ["sample_id", "relative_path", "split", "group", "source_id", "label", "is_wake_word",
                   "category", "affected_word", "n_events", "events_summary", "duration_seconds", "speaker_kind",
                   "provenance", "generation_method", "context", "augmentation", "transcript", "fluent_output",
                   "random_seed", "stutter_profile", "emotion", "age_group", "secondary_behaviour"]


# ----------------------------------------------------------------------------- neutral names
# One place that decides how sources are named in everything this module writes (manifest, annotations,
# summaries, source audit). No third-party application or tool names reach the dataset files or the UI.
# Order matters: longer / more specific patterns first. The mapping is one-to-one, so group-based
# splitting and cross-validation are unaffected.
_NEUTRAL = [
    (re.compile(r"whatsapp[ _-]*audio", re.I), "recording"),
    (re.compile(r"whatsapp", re.I), "recording"),
    (re.compile(r"piper_tts"), "neural_tts"),
    (re.compile(r"PIPERNEG_"), "NTTSNEG_"),
    (re.compile(r"PIPER_"), "NTTS_"),
    (re.compile(r"piperneg:"), "ttsneg:"),
    (re.compile(r"piper:"), "tts:"),
    (re.compile(r"real_stutter_sep28k"), "real_stutter_clips"),
    (re.compile(r"(child|teen|elderly)_sim"), r"\1"),
    (re.compile(r"synthetic", re.I), "generated"),
]


def neutral(value):
    """Return `value` with third-party names replaced; non-strings are returned unchanged."""
    if not isinstance(value, str):
        return value
    for rx, new in _NEUTRAL:
        value = rx.sub(new, value)
    return value


def neutralise_existing(data_dir=DATA_DIR, model_dir=MODEL_DIR, log=print):
    """Rename the third-party names inside an ALREADY BUILT dataset and its result files, in place.
    Run once:  python -m wakeword.dataset_builder      (a .bak copy of every changed file is kept)."""
    n = 0
    for base, pats in ((Path(data_dir), ("*.csv", "*.json", "*.jsonl")), (Path(model_dir), ("*.json", "*.md"))):
        if not base.exists():
            continue
        for pat in pats:
            for f in sorted(base.rglob(pat)):
                if f.suffix == ".bak" or "audio" in f.relative_to(base).parts:
                    continue
                old = f.read_text(encoding="utf-8", newline="")
                new = neutral(old)
                if new != old:
                    f.with_name(f.name + ".bak").write_text(old, encoding="utf-8", newline="")
                    f.write_text(new, encoding="utf-8", newline="")
                    log(f"renamed names in {f}")
                    n += 1
    log(f"done: {n} file(s) changed" if n else "nothing to change: no third-party names found")
    return n


# ----------------------------------------------------------------------------- splitting
def assign_splits(groups_by_kind: dict, rng: np.random.Generator, fractions=(0.7, 0.15, 0.15)) -> dict:
    """Group-wise split (a speaker / recording never appears in two splits)."""
    out = {}
    for kind, groups in groups_by_kind.items():
        groups = sorted(set(groups))
        order = list(rng.permutation(len(groups)))
        n = len(groups)
        if n == 1:
            out[groups[0]] = "train"
            continue
        if n == 2:
            out[groups[order[0]]] = "train"
            out[groups[order[1]]] = "test"
            continue
        n_val = max(1, int(round(n * fractions[1])))
        n_test = max(1, int(round(n * fractions[2])))
        for r, gi in enumerate(order):
            out[groups[gi]] = "test" if r < n_test else ("val" if r < n_test + n_val else "train")
    return out


def _audio(x):
    """Pool items are arrays or lazy loaders (real-speech clips are read only when they are used)."""
    if callable(x):
        x = x()
    return None if x is None else sep28k.as_float(x)


def assign_splits_weighted(weights: dict, rng: np.random.Generator, fractions=(0.7, 0.15, 0.15)) -> dict:
    """group -> split so that the CLIP shares (not the group counts) match `fractions`.  A few huge groups (one show
    can hold thousands of clips) would otherwise make val or test far larger or smaller than planned."""
    names, tot = ("train", "val", "test"), float(sum(weights.values()))
    got, out = dict.fromkeys(names, 0.0), {}
    for g in sorted(weights, key=lambda k: (-weights[k], float(rng.random()))):
        sp = max(zip(names, fractions), key=lambda nf: nf[1] * tot - got[nf[0]])[0]
        out[g] = sp
        got[sp] += weights[g]
    return out


_AGE_LABEL = {"child_sim": "child", "teen_sim": "teen", "elderly_sim": "elderly", "adult": "adult"}


def _speaker_labels(src) -> tuple:
    """(emotion, age_group) of a source voice; recordings of real people carry no such label."""
    if src.kind == "real":
        return "unspecified", "unspecified"
    m = getattr(src.utt, "meta", None) or {}
    age = str(m.get("age") or "unspecified")
    return str(m.get("emotion") or "unspecified"), _AGE_LABEL.get(age, age)


NOISE_SPLITS = ["train"] * 14 + ["val"] * 3 + ["test"] * 3        # fixed 70 / 15 / 15 so the split sizes are known in advance


# ----------------------------------------------------------------------------- helpers
def _events_summary(events):
    return ";".join(f"{e['type']}:{e['word']}:{e.get('unit', '')}:{e['start']}-{e['end']}" for e in events) or "none"


def _add_context(syn, ctx_clip, where, rng, noise_ref):
    """Surround a synthesised wake word with unrelated speech (e.g. the command after it)."""
    pieces = []
    off = 0.0
    if where in ("pre", "both"):
        pre = external_pieces(ctx_clip[: int(1.8 * SR)])
        gap = Piece(dsp_noise(noise_ref, rng.uniform(120, 500), rng), 0, "gap")
        pieces += pre + [gap]
    core = Piece(syn.audio, 0, "kw")
    pieces.append(core)
    if where in ("post", "both"):
        post_clip = ctx_clip[-int(1.8 * SR):] if where == "post" else ctx_clip[: int(1.2 * SR)]
        pieces += [Piece(dsp_noise(noise_ref, rng.uniform(100, 450), rng), 0, "gap")] + external_pieces(post_clip)
    audio, placed = _flatten(pieces, xf_ms=4.0)
    segs, shift = [], 0.0
    for p, (a, b) in placed:
        if p.tag == "kw":
            shift = a / SR
            for s, e, c, t in syn.segments:
                segs.append((s + shift, e + shift, c, t))
        else:
            segs.append((a / SR, b / SR, p.cls, p.tag))
    events = [dict(e, start=round(e["start"] + shift, 3), end=round(e["end"] + shift, 3)) for e in syn.events]
    return audio, segs, events


def dsp_noise(ref, ms, rng):
    from . import dsp
    return dsp.make_noise_floor(ref, int(SR * ms / 1000), rng)


class _Writer:
    def __init__(self, out_dir: Path):
        self.out = Path(out_dir)
        self.rows, self.ann = [], []
        self.counter = Counter()

    def add(self, prefix, folder, x, split, group, source_id, label, is_wake, category, segments, events,
            speaker_kind, is_synth, method, seed, aug=None, context="none", affected="none", fluent_output="",
            profile="none", emotion="n/a", age="n/a", secondary="none"):
        self.counter[prefix] += 1
        sid = f"{prefix}_{self.counter[prefix]:05d}"
        rel = f"audio/{folder}/{sid}.wav"
        audio_io.save_wav(self.out / rel, x)
        label, category, speaker_kind, method = neutral(label), neutral(category), neutral(speaker_kind), neutral(method)
        group, source_id = neutral(group), neutral(source_id)
        self.rows.append({
            "sample_id": sid, "relative_path": rel, "split": split, "group": group, "source_id": source_id,
            "label": label, "is_wake_word": int(is_wake), "category": category, "affected_word": affected,
            "n_events": len(events), "events_summary": _events_summary(events),
            "duration_seconds": round(len(x) / SR, 3), "speaker_kind": speaker_kind,
            "provenance": "generated" if is_synth else "recorded", "generation_method": method,
            "context": context, "augmentation": json.dumps(aug or {}, separators=(",", ":")),
            "transcript": "Hey Google" if is_wake else "(not the wake word)",
            "fluent_output": fluent_output, "random_seed": seed, "stutter_profile": profile,
            "emotion": emotion, "age_group": age, "secondary_behaviour": secondary})
        self.ann.append({"sample_id": sid, "split": split, "is_wake_word": int(is_wake), "category": category,
                         "segments": [[round(a, 4), round(b, 4), int(c), t] for a, b, c, t in segments],
                         "events": events})
        return sid

    def finish(self):
        with open(self.out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            w.writerows(self.rows)
        with open(self.out / "annotations.jsonl", "w", encoding="utf-8") as fh:
            for a in self.ann:
                fh.write(json.dumps(a) + "\n")
        summ = defaultdict(lambda: defaultdict(int))
        for r in self.rows:
            summ[r["split"]][f"{r['label']}/{r['category']}"] += 1
            summ[r["split"]]["_total"] += 1
        with open(self.out / "summary.json", "w", encoding="utf-8") as fh:
            json.dump({k: dict(v) for k, v in summ.items()}, fh, indent=2, sort_keys=True)
        return summ


# ----------------------------------------------------------------------------- main entry
# How many clips are produced per source recording (real recorded voices are the target domain,
# so they are expanded much more than the clean synthetic voices).
# v1.2: fluent and stuttered clips are made 1:1 per source (v1.1 made 8:20 / 2:4, so the models learnt that the
# wake word "sounds stuttered" and missed fluent 'Hey Google').  More hard negatives per voice, because the false
# accepts of v1.1 were dominated by 'Hey <word>', '<word> Google' and near-miss phrases.
# v1.2b: real recorded voices are the target domain (a live microphone) -> 30 + 30 clips per real source.
PLAN = {"real": dict(n_fluent=30, n_dis=30, n_hard=8), "tts": dict(n_fluent=4, n_dis=4, n_hard=3)}
TEAM_AUG_COPIES = 2          # extra channel-augmented copies of every recorded-style stuttered clip (train split only)
IGNORE_CATS = {"near_miss", "hard_google", "partial", "other_wake"}   # look-alike words: no frame labels
PIPER_NEG = {"speech": 450, "near_miss": 1000, "hard_google": 500, "other_wake": 450, "partial": 300, "noise": 180}


def build_dataset(out_dir=DATA_DIR, raw_dir=RAW_DIR, seed: int = 1337, scale: float = 1.0, extra_sources=None,
                  n_piper: int = 1000, n_torgo_per_speaker: int = 40, n_noise: int = 180, refiner=None, extra_audit=None, sep28k_root=None, n_sep: int = 0, pos_scale: float = 1.0, pos_share: float = 0.20, log=print):
    """scale < 1 builds a smaller dataset (quick tests); extra_sources = verified new recordings.

    pos_scale multiplies the number of fluent / stuttered clips made per source recording.
    pos_share is the share of wake-word clips in EVERY split (train, val and test alike); the real-speech negatives are
    trimmed per split to reach it.  n_sep = 0 picks the number of real-speech clips automatically."""
    t0 = time.time()
    out_dir, raw_dir = Path(out_dir), Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("audio/*/*.wav"):
        p.unlink()
    rng = np.random.default_rng(seed)
    sc = lambda n: max(1, int(round(n * scale)))

    # ---- 1. fluent sources --------------------------------------------------------
    log("[1/6] loading fluent sources ...")
    srcs, _ = sources.load_curated(raw_dir)
    srcs += list(extra_sources or [])
    srcs += sources.load_tts(raw_dir)
    srcs += sources.load_piper_positives(raw_dir, sc(n_piper), rng)
    if not srcs:
        raise SystemExit(f"No fluent 'Hey Google' sources found in {raw_dir}. Put the .zip files there.")
    if refiner is not None:
        from .align import refine_utterance
        from .sources import _unit_sane
        n_ok = 0
        for s in srcs:
            r = refine_utterance(refiner, s.utt)
            if r is not None and _unit_sane(r) and sources.speech_qa(r)[0]:
                s.utt, s.flagged, n_ok = r, False, n_ok + 1
            elif _unit_sane(s.utt) and sources.speech_qa(s.utt)[0]:
                s.flagged = False          # model alignment failed the speech check: keep the energy boundaries
            else:
                s.flagged = True
        log(f"      forced alignment refined {n_ok}/{len(srcs)} sources")
    else:
        for s in srcs:
            if not s.flagged and not sources.speech_qa(s.utt)[0]:
                s.flagged = True
    qa = Counter(sources.speech_qa(s.utt)[1] for s in srcs if s.flagged)
    if qa:
        log("      rejected sources - why their Hey/Goo/gle labels fail: " + ", ".join(f"{k} {v}" for k, v in qa.most_common()))
    audit = [{"source": s.id, "group": s.group, "kind": s.kind, "status": "REJECTED_bad_boundaries" if s.flagged else "used",
              "origin": s.origin} for s in srcs]
    audit += list(extra_audit or [])
    good = [s for s in srcs if not s.flagged]
    nk = Counter(s.kind for s in good)
    log(f"      {len(good)} usable sources ({len(srcs) - len(good)} rejected: implausible unit boundaries)  {dict(nk)}")

    team = []
    if refiner is not None:
        team = sources.load_team_disfluent(raw_dir, log=log)
    else:
        log("      (no alignment model yet: the recorded-style stuttered clips are added in the final build)")

    log("[2/6] loading negative pools ...")
    torgo = sources.load_torgo_pool(raw_dir, sc(n_torgo_per_speaker), rng)
    pneg = sources.load_piper_negatives(raw_dir, {k: sc(v) for k, v in PIPER_NEG.items()}, rng)
    plan_of = lambda s: PLAN["real" if s.kind == "real" else "tts"]
    est_pos = sum(sc(round(plan_of(s)["n_fluent"] * pos_scale)) + sc(round(plan_of(s)["n_dis"] * pos_scale)) for s in good)
    est_fixed = sum(min(sc(plan_of(s)["n_hard"]), len(HARD_KINDS)) for s in good) + len(torgo) + len(pneg) + sc(n_noise)
    need = max(0, int(round(est_pos * (1 - pos_share) / pos_share)) - est_fixed)
    n_sep_eff = int(n_sep) if n_sep and n_sep > 0 else int(need * 1.25) + 50
    if sep28k_root:                      # the full SEP-28k replaces the small 'clips' sample (same recordings)
        clips = sep28k.load_pool(sep28k_root, n_sep_eff, seed, log=log)
    else:
        clips = sources.load_clips_pool(raw_dir)
    log(f"      dysarthria/control: {len(torgo)}   real-stutter clips: {len(clips)}   generated-voice negatives: {len(pneg)}")

    # ---- 2. group-wise splits -------------------------------------------------------
    gk = defaultdict(list)
    for s in good:
        fam = "real" if s.kind == "real" else ("tts_libritts" if "libritts" in s.group else "tts_single_voice")
        gk[fam].append(s.group)
    for t in team:                                   # groups whose fluent source was rejected still need a split
        fam = "real" if t["kind"] == "real" else "tts_single_voice"
        if t["group"] not in gk[fam]:
            gk[fam].append(t["group"])
    split_of = assign_splits(gk, rng)
    ext_groups = defaultdict(list)
    for g, kind, _ in torgo:
        ext_groups[kind].append(g)
    for g, cat, _, _ in pneg:
        ext_groups["piperneg:" + cat].append(g)
    split_of.update(assign_splits(ext_groups, rng))
    split_of.update(assign_splits_weighted(Counter(g for g, _, _ in clips), rng))      # real speech: by clip share
    pool = defaultdict(list)                     # unrelated speech per split: used as context + babble noise
    for g, kind, x in torgo:
        pool[split_of[g]].append(x)
    n_ctx = Counter()
    for g, kind, x in clips:                     # a few hundred clips per split are enough as context / babble
        if n_ctx[split_of[g]] < 400:
            a = _audio(x)
            if a is not None:
                pool[split_of[g]].append(a)
                n_ctx[split_of[g]] += 1
    for g, cat, x, _ in pneg:
        if cat == "speech":
            pool[split_of[g]].append(x)

    W = _Writer(out_dir)

    # ---- 3. positives + hard negatives ------------------------------------------------
    log("[3/6] synthesising fluent / stuttered wake words and hard negatives ...")
    for si, src in enumerate(good):
        split = split_of[src.group]
        ctx_pool = pool[split]
        plan = PLAN["real" if src.kind == "real" else "tts"]
        nf, nd, nh = sc(round(plan["n_fluent"] * pos_scale)), sc(round(plan["n_dis"] * pos_scale)), sc(plan["n_hard"])
        emo, age = _speaker_labels(src)
        for k in range(nf + nd):
            is_fluent = k < nf
            sseed = int(rng.integers(0, 2**31 - 1))
            r = np.random.default_rng(sseed)
            u = augment.perturb_utterance(src.utt, r, wide=src.kind == "real") if (k > 0 and r.random() < 0.9) else src.utt
            profile = "none"
            if is_fluent:
                spec = {"type": "fluent"}
            else:                                  # the stuttering profile decides which kind of disfluency this clip gets
                profile = str(r.choice(PROFILE_NAMES, p=PROFILE_WEIGHTS))
                spec = random_spec(r, None, profile, None if emo == "unspecified" else emo, None if age == "unspecified" else age)
            dtype = spec["type"]
            syn = DisfluencySynthesizer(u, r).build(spec)
            audio, segs, events, ctx = syn.audio, syn.segments, syn.events, "none"
            if ctx_pool and r.random() < 0.25:
                where = str(r.choice(["pre", "post", "both"]))
                audio, segs, events = _add_context(syn, ctx_pool[int(r.integers(0, len(ctx_pool)))], where, r, u.noise)
                ctx = where
            y, aug = augment.augment_channel(audio, r, ctx_pool, u.noise)
            sec = sorted({e["type"].split("_", 1)[1] for e in events if e["type"].startswith("secondary_")})
            if float(syn.params.get("tense", 0.0)) >= 0.5:
                sec.append("tense_voicing")
            W.add("FLU" if is_fluent else "DIS", "fluent" if is_fluent else "disfluent", y, split, src.group, src.id,
                  syn.label, True, dtype, segs, events, "real_recording" if src.kind == "real" else "tts_voice",
                  src.kind != "real" or not is_fluent, "unit_resynthesis" if is_fluent else "wsola_subword_splicing",
                  sseed, {**aug, **{k2: v for k2, v in u.meta.items() if k2 in ("tempo", "semitones")}}, ctx,
                  syn.affected_word, "Hey Google", profile, emo, age, ",".join(sec) or "none")
        for kind in rng.choice(HARD_KINDS, size=min(nh, len(HARD_KINDS)), replace=False):
            sseed = int(rng.integers(0, 2**31 - 1))
            r = np.random.default_rng(sseed)
            u = augment.perturb_utterance(src.utt, r, wide=src.kind == "real") if r.random() < 0.6 else src.utt
            syn = negatives.hard_negative(u, str(kind), r, ctx_pool)
            y, aug = augment.augment_channel(syn.audio, r, ctx_pool, u.noise)
            W.add("NEG", "negative", y, split, src.group, src.id, "negative", False, f"hard:{kind}", syn.segments, [],
                  "real_recording" if src.kind == "real" else "tts_voice", True, "unit_recombination", sseed, aug,
                  emotion=emo, age=age)
        if (si + 1) % 100 == 0:
            log(f"      {si + 1}/{len(good)} sources done ({time.time() - t0:.0f}s)")

    # ---- 3b. recorded-style stuttered clips of the project datasets (forced-aligned labels) ---------------
    if team:
        from .align import align_keyword_clip
        log(f"[3b] aligning {len(team)} recorded-style stuttered clips ...")
        n_ok, n_bad = 0, 0
        for t in team:
            res = align_keyword_clip(refiner, t["x"])
            if res is None:
                n_bad += 1
                continue
            lab, (a, b) = res
            split = split_of[t["group"]]
            segs = labels_to_segments(lab, tag="aligned")
            ev = [{"type": t["dtype"], "word": t["word"], "unit": "recorded", "start": round(a * 0.01, 3),
                   "end": round(b * 0.01, 3)}]
            kind = "real_recording" if t["kind"] == "real" else "tts_voice"
            ctx_pool = pool[split]
            for c in range(1 + (TEAM_AUG_COPIES if split == "train" else 0)):
                sseed = int(rng.integers(0, 2**31 - 1))
                r = np.random.default_rng(sseed)
                y, aug = (t["x"], {}) if c == 0 else augment.augment_channel(t["x"], r, ctx_pool, None)
                W.add("DIS", "disfluent", y, split, t["group"], t["group"], "disfluent", True, t["dtype"], segs, ev,
                      kind, True, "recorded_splicing", sseed, aug, "project_dataset", t["word"], "Hey Google",
                      "project_dataset", "unspecified", "unspecified", "none")
            n_ok += 1
        log(f"      recorded-style stuttered clips used: {n_ok}  (alignment implausible, skipped: {n_bad})")

    # ---- 4. external speech negatives -----------------------------------------------------
    log("[4/6] writing external speech negatives ...")
    SPLITS = ("train", "val", "test")
    npos, fixed = Counter(), Counter()
    for row in W.rows:
        (npos if row["is_wake_word"] else fixed)[row["split"]] += 1
    for g, _, _ in torgo:
        fixed[split_of[g]] += 1
    for g, _, _, _ in pneg:
        fixed[split_of[g]] += 1
    for i in range(sc(n_noise)):
        fixed[NOISE_SPLITS[i % len(NOISE_SPLITS)]] += 1
    by_split = defaultdict(list)
    for item in clips:
        by_split[split_of[item[0]]].append(item)
    kept = []
    for sp in SPLITS:
        want = int(round(npos[sp] * (1 - pos_share) / pos_share)) - fixed[sp]
        items = by_split[sp]
        if want > len(items):
            log(f"      note: {sp} has {len(items)} real-speech clips for {max(want, 0)} wanted: its wake-word share will be above {pos_share:.0%}"
                + ("" if sep28k_root else "  (pass --sep28k PATH to use your full real-speech clips)"))
        kept += [items[i] for i in rng.permutation(len(items))[:max(0, want)]]
    log(f"      real-speech negatives kept: {len(kept)} of {len(clips)} (wake-word share target {pos_share:.0%} in every split)")
    clips = kept
    for g, kind, x in torgo:
        r = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        y, aug = augment.augment_channel(x, r, pool[split_of[g]], None, strength=0.5)
        W.add("NEG", "negative", y, split_of[g], g, g, "negative", False, f"speech:{kind}",
              labels_to_segments(vad_labels(y)), [], kind, False, "external_corpus", 0, aug)
    for g, kind, x in clips:
        x = _audio(x)
        if x is None:
            continue
        W.add("NEG", "negative", x, split_of[g], g, g, "negative", False, "speech:real_stutter_clips",
              labels_to_segments(vad_labels(x)), [], "real_stuttering", False, "external_corpus", 0)
    for g, cat, x, text in pneg:
        r = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        y, aug = augment.augment_channel(x, r, pool[split_of[g]], None, strength=0.6)
        lab = vad_labels(y)
        if cat in IGNORE_CATS:                         # e.g. 'Hey Poodle': acoustically close to the keyword, so
            lab = np.where(lab == OTHER, IGNORE, lab)    # its frames carry no label (the verifier still sees it)
        W.add("NEG", "negative", y, split_of[g], g, g, "negative", False, f"tts:{cat}",
              labels_to_segments(lab), [], "tts_voice", True, "neural_tts", 0, {**aug, "text": text})

    # ---- 5. non-speech negatives ------------------------------------------------------------
    log("[5/6] synthesising noise negatives ...")
    refs = [s.utt.noise for s in good if s.kind == "real" and len(s.utt.noise) > 1600]
    for i in range(sc(n_noise)):
        kind = NOISE_KINDS[i % len(NOISE_KINDS)]
        sseed = int(rng.integers(0, 2**31 - 1))
        r = np.random.default_rng(sseed)
        ref = refs[int(r.integers(0, len(refs)))] if refs else None
        x, lab = negatives.noise_negative(kind, r, ref)
        split = NOISE_SPLITS[i % len(NOISE_SPLITS)]
        y = np.clip(x * float(10 ** (r.uniform(-10, 6) / 20)), -0.97, 0.97).astype(np.float32)
        W.add("NEG", "negative", y, split, f"noise_{kind}", f"noise_{kind}", "negative", False, f"noise:{kind}",
              labels_to_segments(vad_labels(y)), [], "noise", True, "synthetic_noise", sseed)

    # ---- 6. manifests --------------------------------------------------------------------------
    log("[6/6] writing manifest ...")
    summ = W.finish()
    with open(out_dir / "source_audit.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "group", "kind", "status", "origin"])
        w.writeheader()
        w.writerows([{k: neutral(v) for k, v in a.items()} for a in audit])
    log(f"done in {time.time() - t0:.0f}s -> {out_dir}")
    for sp in ("train", "val", "test"):
        d = summ.get(sp, {})
        pos = sum(v for k, v in d.items() if k.startswith(("fluent/", "disfluent/")))
        tot = d.get("_total", 0)
        log(f"      {sp:5s} total={tot:5d}  wake-word={pos:5d} ({100 * pos / max(tot, 1):.1f}%)  negatives={tot - pos:5d}")
    return summ


# ----------------------------------------------------------------------------- audit
_FORBIDDEN = re.compile(r"whatsapp|piper|sep-?28k|synthetic|_sim\b", re.I)


def audit_dataset(data_dir=DATA_DIR, log=print, check_audio: int = 150) -> int:
    """Check a built dataset and print one line per check.  Returns the number of FAILED checks (WARN is advice only)."""
    data_dir = Path(data_dir)
    mf = data_dir / "manifest.csv"
    if not mf.exists():
        log(f"[audit] FAIL  no manifest at {mf}  (run: python run.py build-data)")
        return 1
    with open(mf, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    res = []

    def add(status, name, detail=""):
        res.append(status)
        log(f"[audit] {status:4s}  {name}{'  ' + detail if detail else ''}")

    missing = [c for c in MANIFEST_FIELDS if rows and c not in rows[0]]
    add("FAIL" if missing else "PASS", "all label columns present", f"missing: {missing}" if missing else f"{len(MANIFEST_FIELDS)} columns, {len(rows)} clips")
    if missing:
        return 1
    bad = [r["sample_id"] for r in rows if (r["label"] in ("fluent", "disfluent")) != (r["is_wake_word"] == "1")]
    add("FAIL" if bad else "PASS", "wake-word flag agrees with the label", f"{len(bad)} contradictions, e.g. {bad[:3]}" if bad else "")
    nodis = [r["sample_id"] for r in rows if r["label"] == "disfluent" and r["n_events"] == "0"]
    flu = [r["sample_id"] for r in rows if r["label"] == "fluent" and r["n_events"] != "0"]
    add("FAIL" if nodis or flu else "PASS", "every stuttered clip has events, no fluent clip has any", f"{len(nodis)} stuttered without events, {len(flu)} fluent with events" if nodis or flu else "")
    g2s = defaultdict(set)
    for r in rows:
        if r["speaker_kind"] != "noise":
            g2s[r["group"]].add(r["split"])
    leak = [g for g, s in g2s.items() if len(s) > 1]
    add("FAIL" if leak else "PASS", "no recording / voice / show appears in two splits", f"{len(leak)} leaking groups, e.g. {leak[:3]}" if leak else f"{len(g2s)} groups, each in one split")
    shares = {}
    for sp in ("train", "val", "test"):
        rs = [r for r in rows if r["split"] == sp]
        shares[sp] = sum(r["is_wake_word"] == "1" for r in rs) / max(len(rs), 1)
    spread = max(shares.values()) - min(shares.values())
    add("PASS" if spread <= 0.03 else "WARN", "same wake-word share in every split",
        "  ".join(f"{k} {100 * v:.1f}%" for k, v in shares.items()) + f"  (spread {100 * spread:.1f} points)")
    for sp in ("train", "val", "test"):
        pos = [r for r in rows if r["split"] == sp and r["is_wake_word"] == "1"]
        fl = sum(r["label"] == "fluent" for r in pos) / max(len(pos), 1)
        add("PASS" if 0.25 <= fl <= 0.6 else "WARN", f"fluent and stuttered wake words both well covered in {sp}", f"{len(pos)} wake words, {100 * fl:.0f}% fluent")
    dis = [r for r in rows if r["label"] == "disfluent"]
    unl = [r["sample_id"] for r in dis if r["stutter_profile"] in ("", "none", "n/a")]
    add("FAIL" if unl else "PASS", "every stuttered clip names its stuttering profile", f"{len(unl)} without" if unl else
        ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["stutter_profile"] for r in dis).items())))
    add("PASS", "emotions covered (generated voices)", ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["emotion"] for r in rows if r["is_wake_word"] == "1").items())))
    add("PASS", "age groups covered (generated voices)", ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["age_group"] for r in rows if r["is_wake_word"] == "1").items())))
    gone = [r["relative_path"] for r in rows if not (data_dir / r["relative_path"]).exists()]
    add("FAIL" if gone else "PASS", "every audio file exists", f"{len(gone)} missing, e.g. {gone[:2]}" if gone else "")
    rng = np.random.default_rng(0)
    unread = 0
    for i in rng.permutation(len(rows))[:check_audio]:
        try:
            x = audio_io.load_audio(str(data_dir / rows[int(i)]["relative_path"]))
            unread += int(len(x) < 1000 or not np.all(np.isfinite(x)))
        except Exception:
            unread += 1
    add("FAIL" if unread else "PASS", f"a random sample of {min(check_audio, len(rows))} files decodes cleanly", f"{unread} unreadable" if unread else "")
    hits = Counter()
    for name in ("manifest.csv", "annotations.jsonl", "summary.json", "source_audit.csv"):
        f = data_dir / name
        if f.exists():
            for m in _FORBIDDEN.findall(f.read_text(encoding="utf-8")):
                hits[(name, m.lower())] += 1
    add("FAIL" if hits else "PASS", "no third-party names or 'synthetic' in the dataset files", str(dict(hits)) if hits else "")
    add("WARN", "speaker independence", "recordings carry no speaker id: a recording is the unit of separation, so two recordings by one person can sit in different splits")
    nf = sum(s == "FAIL" for s in res)
    log(f"[audit] {'FAILED: ' + str(nf) + ' check(s)' if nf else 'PASSED'}  ({sum(s == 'WARN' for s in res)} advisory warning(s))")
    return nf


if __name__ == "__main__":
    neutralise_existing()

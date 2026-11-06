"""Synthesis of realistic stuttered 'Hey Google' from a speaker's own fluent recording.

Design rules (why the output does not sound like a digital loop):
  * Repetitions are *sub-word progressive attempts* ("g-", "gu-", "goo-", "google"), not copies
    of one clip. Every attempt gets its own cut length, tempo, pitch, gain and fade.
  * Silences / blocks are filled with the *recording's own background noise* (randomly tiled),
    never with digital zero, unless the source itself is digitally silent.
  * Prolongations stretch only the steady part of a vowel (or the /h/ aspiration) with WSOLA
    plus slow jitter, a mild tremor and a slow decay, so a held sound is not a perfect loop.
  * Because everything is cut from the speaker's own audio, the voice, accent, microphone and
    WhatsApp codec colouring of that speaker are preserved ("stuttering in their voice").

Every synthesised clip carries exact frame-level annotations (segments) and an event list.
Parameter ranges are literature-informed priors; they are NOT calibrated on SEP-28k /
FluencyBank labels yet (see README, 'Honest limitations').
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dsp
from .config import BG, GLE, GOO, HEY, HOP, IGNORE, SR, WIN
from .features import n_frames
from .segmenter import Utterance

DISFLUENCY_TYPES = ["sound_repetition", "syllable_repetition", "word_repetition",
                    "sound_prolongation", "final_prolongation", "block_interruption",
                    "interjection", "revision", "mixed"]
# interjection = filled pause ('um / uh / eh') before or inside the phrase  (SEP-28k 'Interjection')
# revision     = an aborted start that is restarted: 'Hey Goo- ... Hey Google'  (Shriberg-style repair)

# ----------------------------------------------------------------------------- stuttering profiles
# A profile is a set of tendencies for HOW a clip is disfluent: which kinds of events are likely, where they fall,
# how tense the voice is, and how often audible secondary behaviours appear.  The tendencies follow the commonly
# described features of each kind of stuttering; they are modelling assumptions of this generator, not measurements,
# and have not been validated by speech-language pathologists.  Eye blinks, grimaces and head nods leave no trace
# in audio, so only audible secondary behaviours (a gasp before the word, tapping, tense voicing) are produced.
PROFILES = {
    # starts in childhood; sounds and syllables at the START of words, tense, with audible effort
    "developmental": dict(weight=0.50, hey=0.55, p_secondary=0.40, tension=0.45, gap=1.0, block=1.0, prolong=1.0,
                          types={"sound_repetition": .20, "syllable_repetition": .22, "word_repetition": .08,
                                 "sound_prolongation": .18, "final_prolongation": .02, "block_interruption": .14, "mixed": .16,
                                 "interjection": .07, "revision": .05}),
    # after a brain injury or illness: any position in the word (also the final syllable), steady rhythm, little tension
    "neurogenic": dict(weight=0.30, hey=0.40, p_secondary=0.10, tension=0.10, gap=0.85, block=0.8, prolong=1.0,
                       types={"sound_repetition": .08, "syllable_repetition": .14, "word_repetition": .20,
                              "sound_prolongation": .14, "final_prolongation": .22, "block_interruption": .06, "mixed": .16,
                              "interjection": .07, "revision": .09}),
    # abrupt onset linked to stress: repetitions and long, effortful blocks, many secondary behaviours, high variability
    "psychogenic": dict(weight=0.20, hey=0.60, p_secondary=0.60, tension=0.70, gap=1.1, block=1.3, prolong=0.9,
                        types={"sound_repetition": .26, "syllable_repetition": .22, "word_repetition": .06,
                               "sound_prolongation": .06, "final_prolongation": .02, "block_interruption": .20, "mixed": .18,
                               "interjection": .08, "revision": .05}),
}
PROFILE_NAMES = list(PROFILES)
PROFILE_WEIGHTS = [PROFILES[k]["weight"] for k in PROFILE_NAMES]
_TENSE_EMO = {"scared": 0.25, "angry": 0.25, "excited": 0.20, "grumpy": 0.15}
_SLOW_EMO = {"sad", "tired"}


def style(profile: str, emotion: str | None, age: str | None) -> dict:
    """Profile tendencies, shifted by the speaker's emotion and age group (a modelling assumption: tense or
    aroused speech is more forced and faster, low-energy speech slower with longer holds, children repeat more)."""
    P = PROFILES[profile]
    s = dict(tension=P["tension"], gap=P["gap"], block=P["block"], prolong=P["prolong"], p_secondary=P["p_secondary"], rep_bonus=0)
    emo, age = (emotion or "").lower(), (age or "").lower()
    if emo in _TENSE_EMO:
        s["tension"] += _TENSE_EMO[emo]; s["gap"] *= 0.85; s["p_secondary"] += 0.10
    elif emo in _SLOW_EMO:
        s["tension"] -= 0.10; s["gap"] *= 1.2; s["prolong"] *= 1.25
    if age == "child":
        s["rep_bonus"] = 1; s["block"] *= 0.8; s["prolong"] *= 0.9; s["tension"] -= 0.10
    elif age == "elderly":
        s["prolong"] *= 1.2; s["block"] *= 1.1; s["gap"] *= 1.1
    s["tension"] = float(np.clip(s["tension"], 0.0, 1.0))
    s["p_secondary"] = float(np.clip(s["p_secondary"], 0.0, 0.9))
    return s
WORDS = ["Hey", "Google"]

_CAL_CACHE: dict = {}


def cal_params() -> dict:
    """Synthesis parameters. Defaults are literature-informed priors; if calibration.json exists
    (python run.py sep28k-calibrate ...) they are replaced by values measured on real stuttered speech."""
    if "p" in _CAL_CACHE:
        return _CAL_CACHE["p"]
    p = {"gap_scale": 55.0, "prolong_ms": (300.0, 1700.0), "block_ms": (260.0, 1800.0), "calibrated": False}
    try:
        from .sep28k import load_calibration
        c = load_calibration()
        g = (c.get("repetition_gap_ms") or {}).get("median")
        if g:
            p["gap_scale"] = float(np.clip(g / 2.67, 30.0, 120.0))
            p["calibrated"] = True
        pr, fl = c.get("prolongation_run_ms") or {}, (c.get("fluent_run_ms") or {}).get("median") or 250.0
        if pr.get("p10") and pr.get("p90"):
            lo, hi = max(250.0, pr["p10"] - fl), min(2500.0, pr["p90"] - fl)
            if hi > lo + 200:
                p["prolong_ms"] = (lo, hi)
                p["calibrated"] = True
        bl = c.get("block_gap_ms") or {}
        if bl.get("p10") and bl.get("p90"):
            lo, hi = max(250.0, bl["p10"]), min(2500.0, bl["p90"])
            if hi > lo + 200:
                p["block_ms"] = (lo, hi)
                p["calibrated"] = True
    except Exception:
        pass
    _CAL_CACHE["p"] = p
    return p


@dataclass
class Piece:
    audio: np.ndarray
    cls: int
    tag: str = ""
    ev: int = -1          # index into events (-1 = none)


@dataclass
class Synth:
    audio: np.ndarray
    segments: list        # [(start_s, end_s, cls, tag)]
    events: list          # [{type, word, unit, start, end, ...}]
    label: str
    disfluency_type: str
    affected_word: str
    params: dict = field(default_factory=dict)
    # canonical fluent rendition spans (seconds) of the *final* Hey / Goo / gle pieces
    fluent_spans: dict = field(default_factory=dict)

    def frame_labels(self) -> np.ndarray:
        return segments_to_frame_labels(self.segments, len(self.audio))


def segments_to_frame_labels(segments, n_samples: int) -> np.ndarray:
    T = n_frames(n_samples)
    lab = np.full(T, BG, dtype=np.int64)
    centers = (np.arange(T) * HOP + WIN // 2) / SR
    for s, e, c, *_ in segments:
        lab[(centers >= s) & (centers < e)] = c
    return lab


# ----------------------------------------------------------------------------- low level helpers
def _cut(x, a, b, rng, jitter_ms=8.0, sr=SR):
    j = int(sr * jitter_ms / 1000)
    a2 = dsp.snap_zero(x, a + int(rng.integers(-j, j + 1)))
    b2 = dsp.snap_zero(x, b + int(rng.integers(-j, j + 1)))
    if b2 - a2 < 80:
        a2, b2 = a, max(b, a + 80)
    return x[max(a2, 0): min(b2, len(x))].astype(np.float32)


def _flatten(pieces, xf_ms=5.0):
    pieces = [p for p in pieces if len(p.audio) > 0]
    if not pieces:
        return np.zeros(0, dtype=np.float32), []
    xf = int(SR * xf_ms / 1000)
    ov = [0]
    for a, b in zip(pieces[:-1], pieces[1:]):
        ov.append(int(min(xf, len(a.audio) // 2, len(b.audio) // 2)))
    starts, pos = [], 0
    for i, p in enumerate(pieces):
        pos = pos - ov[i] if i else 0
        starts.append(pos)
        pos += len(p.audio)
    total = pos
    out = np.zeros(total, dtype=np.float32)
    for i, p in enumerate(pieces):
        a = p.audio.astype(np.float32).copy()
        if ov[i] > 1:
            a[:ov[i]] *= np.sin(np.linspace(0, np.pi / 2, ov[i], dtype=np.float32))
        nxt = ov[i + 1] if i + 1 < len(pieces) else 0
        if nxt > 1:
            a[-nxt:] *= np.cos(np.linspace(0, np.pi / 2, nxt, dtype=np.float32))
        out[starts[i]: starts[i] + len(a)] += a
    spans = [(starts[i], starts[i] + len(p.audio)) for i, p in enumerate(pieces)]
    return out, list(zip(pieces, spans))


# ----------------------------------------------------------------------------- the synthesiser
class DisfluencySynthesizer:
    """Builds fluent / disfluent variants of one speaker utterance."""

    def __init__(self, u: Utterance, rng: np.random.Generator):
        self.u = u
        self.rng = rng
        self.x = u.audio
        self.s, self.vo, self.b1, self.b2, self.e = u.start, u.vowel_onset, u.b1, u.b2, u.end
        self.utt_rms = dsp.rms(self.x[self.s:self.e]) + 1e-6
        self.events: list = []
        self.tense = 0.0

    # ---- primitives -------------------------------------------------------------
    def noise(self, ms: float) -> np.ndarray:
        n = int(SR * ms / 1000)
        return dsp.make_noise_floor(self.u.noise, n, self.rng)

    def gap(self, ms: float, tag="gap", ev=-1, breath_p=0.0) -> list:
        parts = [Piece(self.noise(ms), BG, tag, ev)]
        if breath_p > 0 and self.rng.random() < breath_p and ms > 260:
            bl = int(self.rng.uniform(120, 260))
            br = dsp.breath(int(SR * bl / 1000), self.rng, level_db=-30) * self.utt_rms * self.rng.uniform(0.25, 0.7)
            # replace the tail of the pause by the breath (an audible gasp right before release)
            parts = [Piece(self.noise(ms - bl), BG, tag, ev), Piece(br, BG, "breath", ev)]
        return parts

    def inhale(self) -> list:
        """An audible gasp before the word, then a short pause."""
        r = self.rng
        n = int(SR * r.uniform(180, 420) / 1000)
        br = dsp.breath(n, r, level_db=-30) * self.utt_rms * r.uniform(0.3, 0.9)
        return [Piece(br.astype(np.float32), BG, "inhale", -2), Piece(self.noise(float(r.uniform(30, 120))), BG, "gap", -2)]

    def taps(self) -> list:
        """One to three dull taps (a fist or foot on a surface) that spill into the recording."""
        r = self.rng
        out = []
        for _ in range(int(r.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))):
            n = int(SR * r.uniform(0.04, 0.09))
            body = np.convolve(r.standard_normal(n).astype(np.float32), np.ones(24, dtype=np.float32) / 24, mode="same")
            body = body * np.exp(-(np.arange(n) / SR) / r.uniform(0.012, 0.03))
            body = body / (np.max(np.abs(body)) + 1e-6) * self.utt_rms * r.uniform(0.5, 1.4)
            out += [Piece(body.astype(np.float32), BG, "tap", -2), Piece(self.noise(float(r.uniform(120, 350))), BG, "gap", -2)]
        return out

    def unit(self, name: str):
        if name == "hey":
            return _cut(self.x, self.s, self.b1, self.rng, 4), HEY
        if name == "goo":
            return _cut(self.x, self.b1, self.b2, self.rng, 4), GOO
        return _cut(self.x, self.b2, self.e, self.rng, 4), GLE

    def _vary(self, seg, k, n, tempo_range=(0.88, 1.15), pitch_sd=1.0, final=False):
        """Acoustic variation of one attempt."""
        r = self.rng
        if len(seg) > 960:
            seg = dsp.wsola(seg, 1.0 / r.uniform(*tempo_range), rng=r)
        if len(seg) > 800 and r.random() < 0.8:
            st = float(np.clip(r.normal(0.35 * k + 1.6 * self.tense, pitch_sd), -2.5, 4.0))
            seg = dsp.pitch_shift(seg, st)
        g = dsp.db_to_gain(r.uniform(-2.5, 2.0) + 0.5 * k + 3.0 * self.tense)
        seg = seg * g
        return dsp.fade(seg, r.uniform(2, 5), 0 if final else r.uniform(8, 25))

    def filler(self, ev: int) -> list:
        """A filled pause ('eh / uh / um') made from the speaker's OWN vowel: the nucleus of 'Hey', held 180-550 ms,
        lowered in pitch and level, optionally closed by a nasal murmur ('um').  Its frames are labelled IGNORE:
        they are not supervised for the frame model (the same vowel is a legitimate 'Hey' in a prolongation), the
        grammar absorbs them as a pause and the verifier learns from whole clips."""
        r = self.rng
        span = self.b1 - self.vo
        lo, hi = self.vo + int(0.25 * span), self.b1 - int(0.25 * span)
        if hi - lo < int(0.04 * SR):
            mid = (self.vo + self.b1) // 2
            lo, hi = mid - int(0.025 * SR), mid + int(0.025 * SR)
        lo, hi = dsp.snap_zero(self.x, max(lo, 0)), dsp.snap_zero(self.x, hi)
        if hi - lo < 400:
            hi = lo + 400
        seg = self.x[lo:hi].astype(np.float32)
        target = int(SR * r.uniform(0.18, 0.55))
        st = dsp.wsola(seg, target / max(len(seg), 1), rng=r, jitter_ms=3.0)
        if len(st) > 800:
            st = dsp.pitch_shift(st, float(r.uniform(-3.0, -0.5)))
        if r.random() < 0.4 and len(st) > 1600:                     # 'um': low-passed nasal murmur at the end
            k = int(len(st) * r.uniform(0.3, 0.5))
            murmur = np.convolve(st[-k:], np.ones(24, dtype=np.float32) / 24, mode="same") * 0.7
            st = dsp.crossfade_concat([st[:-k], murmur.astype(np.float32)], xf_ms=15)
        st = dsp.fade(st * dsp.db_to_gain(r.uniform(-8.0, -3.0)), 25, 60)
        return [Piece(st.astype(np.float32), IGNORE, "filler", ev)]

    def restart(self, ev: int) -> list:
        """Aborted first attempt 'Hey Goo-' (complete Hey, part of Goo), a pause, then the caller's full phrase."""
        r = self.rng
        hey_a, _ = self.unit("hey")
        hey_a = self._vary(hey_a, 0, 1)
        b = self.b1 + int((self.b2 - self.b1) * r.uniform(0.45, 0.8))
        goo_p = self._vary(_cut(self.x, self.b1, max(b, self.b1 + int(0.05 * SR)), r, 6), 0, 1)
        return ([Piece(hey_a, HEY, "attempt", ev)] + self.gap(float(r.uniform(30, 90)), "gap", ev)
                + [Piece(goo_p, GOO, "attempt", ev)] + self.gap(float(r.uniform(150, 600)), "gap", ev, breath_p=0.3))

    # ---- disfluency events (each returns the replacement pieces) -----------------
    def repetition(self, word: str, unit: str, count: int, ev: int) -> list:
        """Progressive attempts followed by the complete word piece(s) (caller adds them)."""
        r = self.rng
        pieces = []
        x = self.x
        for k in range(count):
            if word == "Hey":
                a = self.s
                if unit == "sound":
                    frac = r.uniform(0.0, 1.0)
                    b = self.vo + int(SR * (0.015 + 0.055 * frac) * (1 + 0.12 * k))
                else:
                    b = self.vo + int((self.b1 - self.vo) * min(1.0, r.uniform(0.5, 0.95) + 0.05 * k))
                b = min(b, self.b1 - int(0.01 * SR))
                cls = HEY
            else:
                a = self.b1
                if unit == "sound":
                    b = a + int(SR * r.uniform(0.07, 0.13) * (1 + 0.12 * k))
                else:
                    b = a + int((self.b2 - self.b1) * min(1.0, r.uniform(0.55, 0.95) + 0.05 * k))
                b = min(b, self.b2 - int(0.01 * SR))
                cls = GOO
            seg = _cut(x, a, max(b, a + int(0.04 * SR)), r, 8)
            seg = self._vary(seg, k, count)
            pieces.append(Piece(seg, cls, "attempt", ev))
            # inter-attempt gap: 40-260 ms, tends to shrink as the speaker speeds up
            gms = float(np.clip(r.gamma(3.0, cal_params()["gap_scale"]) * (1.0 - 0.07 * k), 35, 300))
            pieces += self.gap(gms, "gap", ev)
        # shorter gap before the fluent release
        if pieces and pieces[-1].tag == "gap":
            pieces[-1].audio = pieces[-1].audio[: max(int(pieces[-1].audio.size * r.uniform(0.15, 0.7)), int(0.015 * SR))]
        return pieces

    def prolong(self, word: str, mode: str, extra_ms: float, ev: int):
        """Return (pre, stretched, post) pieces of the unit with a prolonged sound."""
        r = self.rng
        if word == "Hey":
            if mode == "onset":
                a, b = self.s, max(self.vo, self.s + int(0.03 * SR))
                lo, hi = a, b
            else:
                span = self.b1 - self.vo
                lo = self.vo + int(0.18 * span)
                hi = self.b1 - int(0.22 * span)
            full_a, full_b, cls = self.s, self.b1, HEY
        elif word == "gle":
            span = self.e - self.b2
            lo = self.b2 + int(0.30 * span)
            hi = self.e - int(0.15 * span)
            full_a, full_b, cls = self.b2, self.e, GLE
        else:
            span = self.b2 - self.b1
            lo = self.b1 + int(0.35 * span)
            hi = self.b2 - int(0.18 * span)
            full_a, full_b, cls = self.b1, self.b2, GOO
        if hi - lo < int(0.035 * SR):
            mid = (full_a + full_b) // 2
            lo, hi = mid - int(0.03 * SR), mid + int(0.03 * SR)
        lo, hi = dsp.snap_zero(self.x, lo), dsp.snap_zero(self.x, hi)
        if hi - lo < 400:
            lo, hi = lo, lo + 400
        seg = self.x[lo:hi].astype(np.float32)
        target = len(seg) + int(SR * extra_ms / 1000)
        st = dsp.wsola(seg, target / len(seg), rng=r, jitter_ms=3.0)
        t = np.arange(len(st)) / SR
        trem = 1.0 + r.uniform(0.0, 0.08) * np.sin(2 * np.pi * r.uniform(4.0, 7.5) * t + r.uniform(0, 6.28))
        decay = np.linspace(1.0, r.uniform(0.65, 1.0), len(st))
        st = (st * trem * decay).astype(np.float32)
        pre = _cut(self.x, full_a, lo, r, 3)
        post = _cut(self.x, hi, full_b, r, 3)
        return (Piece(pre, cls, "unit", ev), Piece(st, cls, "prolong", ev), Piece(post, cls, "unit", ev))

    # ---- assembling ----------------------------------------------------------------
    def build(self, spec: dict) -> Synth:
        r = self.rng
        self.events = []
        dtype = spec.get("type", "fluent")
        slots = {"pre": [], "hey": [], "gap1": [], "goo": [], "gap2": [], "gle": [], "post": []}
        hey_a, hey_c = self.unit("hey")
        goo_a, goo_c = self.unit("goo")
        gle_a, gle_c = self.unit("gle")
        slots["hey"] = [Piece(hey_a, HEY, "unit")]
        slots["goo"] = [Piece(goo_a, GOO, "unit")]
        slots["gle"] = [Piece(gle_a, GLE, "unit")]
        applied = []

        def new_event(**kw):
            self.events.append(kw)
            return len(self.events) - 1

        def apply(kind, word=None, **kw):
            ev_args = dict(kw)
            if kind == "rep":
                unit = kw["unit"]
                cnt = int(kw["count"])
                ev = new_event(type="repetition", word=word, unit=unit, count=cnt)
                att = self.repetition(word, unit, cnt, ev)
                slot = "hey" if word == "Hey" else "goo"
                slots[slot] = att + slots[slot]
                applied.append(("rep", word))
            elif kind == "wordrep":
                cnt = int(kw["count"])
                ev = new_event(type="word_repetition", word=word, unit="word", count=cnt)
                pcs = []
                for k in range(cnt):
                    if word == "Hey":
                        a, c = self.unit("hey")
                        seg = self._vary(a, k, cnt, pitch_sd=0.8)
                        pcs += [Piece(seg, HEY, "attempt", ev)] + self.gap(float(np.clip(r.gamma(3.0, cal_params()["gap_scale"] * 1.1), 50, 380)), "gap", ev)
                    else:
                        a, _ = self.unit("goo")
                        g, _ = self.unit("gle")
                        seg = self._vary(np.concatenate([a, g]), k, cnt, pitch_sd=0.8)
                        pcs += [Piece(seg[: len(a)], GOO, "attempt", ev), Piece(seg[len(a):], GLE, "attempt", ev)]
                        pcs += self.gap(float(np.clip(r.gamma(3.0, cal_params()["gap_scale"] * 1.1), 50, 380)), "gap", ev)
                if word == "Hey":
                    slots["hey"] = pcs + slots["hey"]
                else:   # 'Google Google' : repeat the whole word before the final one
                    slots["goo"] = pcs + slots["goo"]
                applied.append(("wordrep", word))
            elif kind == "prolong":
                mode = kw["mode"]
                extra = float(kw["extra_ms"])
                final = mode == "final"
                ev = new_event(type="prolongation", word="Google" if final else word, unit="final_syllable" if final else mode,
                               duration_ms=round(extra))
                pre, st, post = self.prolong("gle" if final else word, mode, extra, ev)
                slot = "gle" if final else ("hey" if word == "Hey" else "goo")
                # replace the unit piece (keep any repetition attempts that precede it)
                keep = [p for p in slots[slot] if p.tag != "unit"]
                slots[slot] = keep + [pre, st, post]
                applied.append(("prolong", word))
            elif kind == "secondary":
                sub, pos = kw["sub"], kw["pos"]
                ev = new_event(type=f"secondary_{sub}", word="Hey" if pos == "pre" else "Google", unit=pos, duration_ms=0)
                pcs = self.inhale() if sub == "inhale" else self.taps()
                for pc in pcs:
                    pc.ev = ev
                slots[{"pre": "pre", "hey_goo": "gap1", "goo_gle": "gap2"}[pos]] += pcs
                applied.append(("secondary", sub))
            elif kind == "interjection":
                pos = kw["pos"]
                ev = new_event(type="interjection", word="Hey" if pos == "pre" else "Google", unit=pos, duration_ms=0)
                pcs = self.gap(float(r.uniform(40, 200)), "gap", ev) + self.filler(ev) + self.gap(float(r.uniform(40, 220)), "gap", ev)
                self.events[ev]["duration_ms"] = round(1000 * sum(len(p.audio) for p in pcs) / SR)
                slots[{"pre": "pre", "hey_goo": "gap1", "goo_gle": "gap2"}[pos]] += pcs
                applied.append(("interjection", pos))
            elif kind == "revision":
                ev = new_event(type="revision", word="Google", unit="restart", count=1)
                slots["hey"] = self.restart(ev) + slots["hey"]
                applied.append(("revision", "Google"))
            elif kind == "block":
                pos = kw["pos"]
                ms = float(kw["ms"])
                ev = new_event(type="block", word="Hey" if pos == "pre" else "Google", unit=pos, duration_ms=round(ms))
                gp = self.gap(ms, "block", ev, breath_p=0.35)
                slots[{"pre": "pre", "hey_goo": "gap1", "goo_gle": "gap2"}[pos]] += gp
                applied.append(("block", pos))

        self.tense = float(spec.get("tense", 0.0))
        if dtype == "fluent":
            pass
        else:
            for item in spec["events"]:
                apply(**item)

        lead = self.rng.uniform(*spec.get("lead_ms", (120, 650)))
        trail = self.rng.uniform(*spec.get("trail_ms", (150, 800)))
        slots["pre"] = self.gap(lead, "lead") + slots["pre"]
        slots["post"] = self.gap(trail, "trail")
        order = ["pre", "hey", "gap1", "goo", "gap2", "gle", "post"]
        pieces = [p for k in order for p in slots[k]]
        audio, placed = _flatten(pieces)
        segs, ev_t = [], {}
        fluent_spans = {}
        for p, (a, b) in placed:
            segs.append((a / SR, b / SR, p.cls, p.tag))
            if p.ev >= 0:
                lo, hi = ev_t.get(p.ev, (1e9, -1e9))
                ev_t[p.ev] = (min(lo, a / SR), max(hi, b / SR))
        # canonical (last, complete) spans for the 'fluent rendition'
        for key, cls in (("hey", HEY), ("goo", GOO), ("gle", GLE)):
            cand = [(a / SR, b / SR) for p, (a, b) in placed if p.cls == cls and p.tag in ("unit", "prolong")]
            if cand:
                fluent_spans[key] = cand
        events = []
        for i, ev in enumerate(self.events):
            ev = dict(ev)
            lo, hi = ev_t.get(i, (0.0, 0.0))
            ev["start"], ev["end"] = round(lo, 3), round(hi, 3)
            events.append(ev)
        events.sort(key=lambda d: d["start"])
        words = {e["word"] for e in events if not e["type"].startswith("secondary")}
        if dtype == "fluent":
            aff = "none"
        elif len(words) > 1 or dtype == "mixed" and len(words) != 1:
            aff = "both"
        else:
            aff = next(iter(words)) if words else "none"
        return Synth(audio=audio, segments=segs, events=events,
                     label="fluent" if dtype == "fluent" else "disfluent", disfluency_type=dtype,
                     affected_word=aff, params=spec, fluent_spans=fluent_spans)


# ----------------------------------------------------------------------------- random specs
def _rep_count(r, bonus: int = 0):
    n = int(r.choice([1, 2, 3, 4, 5], p=[0.28, 0.30, 0.22, 0.12, 0.08]))
    return min(6, n + (1 if bonus and r.random() < 0.5 else 0))


def random_spec(rng: np.random.Generator, dtype: str | None = None, profile: str = "developmental",
                emotion: str | None = None, age: str | None = None) -> dict:
    """Sample a disfluency specification for a stuttering profile, speaker emotion and age group.
    With dtype=None the profile decides which kind of disfluency the clip gets."""
    if dtype == "fluent":
        return {"type": "fluent"}
    P, S = PROFILES[profile], style(profile, emotion, age)
    if dtype is None:
        names = list(P["types"])
        pr = np.array([P["types"][n] for n in names], dtype=float)
        dtype = str(rng.choice(names, p=pr / pr.sum()))
    gap_lo, gap_hi = cal_params()["prolong_ms"]
    blk_lo, blk_hi = cal_params()["block_ms"]
    pm = lambda lo, hi: float(np.clip(rng.uniform(lo, hi) * S["prolong"], 250, 2800))
    bm = lambda lo, hi: float(np.clip(rng.uniform(lo, hi) * S["block"], 200, 2800))
    word = lambda: "Hey" if rng.random() < P["hey"] else "Google"
    cnt = lambda: _rep_count(rng, S["rep_bonus"])
    ev = []
    if dtype == "sound_repetition":
        ev.append(dict(kind="rep", word=word(), unit="sound", count=cnt()))
    elif dtype == "syllable_repetition":
        ev.append(dict(kind="rep", word=word(), unit="syllable", count=cnt()))
    elif dtype == "word_repetition":
        ev.append(dict(kind="wordrep", word=word(), count=int(rng.choice([1, 2, 3], p=[0.55, 0.3, 0.15]))))
    elif dtype == "sound_prolongation":
        w = word()
        mode = "onset" if (w == "Hey" and rng.random() < 0.3) else "vowel"
        ev.append(dict(kind="prolong", word=w, mode=mode, extra_ms=pm(gap_lo, gap_hi)))
    elif dtype == "final_prolongation":
        ev.append(dict(kind="prolong", word="Google", mode="final", extra_ms=pm(gap_lo, gap_hi)))
    elif dtype == "block_interruption":
        pos = str(rng.choice(["pre", "hey_goo", "goo_gle"], p=[0.15, 0.6, 0.25]))
        ev.append(dict(kind="block", word=None, pos=pos, ms=bm(blk_lo, blk_hi)))
    elif dtype == "interjection":
        ev.append(dict(kind="interjection", word=None, pos=str(rng.choice(["pre", "hey_goo"], p=[0.35, 0.65]))))
    elif dtype == "revision":
        ev.append(dict(kind="revision", word=None))
    elif dtype == "mixed":
        pool = [
            dict(kind="rep", word="Hey", unit=str(rng.choice(["sound", "syllable"])), count=cnt()),
            dict(kind="rep", word="Google", unit=str(rng.choice(["sound", "syllable"])), count=cnt()),
            dict(kind="prolong", word="Hey", mode="vowel", extra_ms=pm(300, 1400)),
            dict(kind="prolong", word="Google", mode="vowel", extra_ms=pm(300, 1400)),
            dict(kind="prolong", word="Google", mode="final", extra_ms=pm(300, 1200)),
            dict(kind="block", word=None, pos="hey_goo", ms=bm(260, 1500)),
            dict(kind="block", word=None, pos="goo_gle", ms=bm(200, 900)),
            dict(kind="interjection", word=None, pos="hey_goo"),
        ]
        n = int(rng.choice([2, 3], p=[0.7, 0.3]))
        pick = rng.choice(len(pool), size=n, replace=False)
        chosen = [pool[i] for i in pick]
        # at most one repetition + one prolongation per word
        seen, ev = set(), []
        for c in chosen:
            key = (c["kind"], c.get("word"), c.get("mode") if c["kind"] == "prolong" else c.get("pos"))
            if key not in seen:
                seen.add(key)
                ev.append(c)
        if len(ev) < 2:
            ev.append(pool[5])
    else:
        raise ValueError(dtype)
    # audible secondary behaviours (a gasp before the word, tapping) and tense voicing
    secondary = []
    if rng.random() < S["p_secondary"]:
        secondary.append(dict(kind="secondary", sub="inhale", pos=str(rng.choice(["pre", "hey_goo"], p=[0.7, 0.3]))))
        if rng.random() < 0.25:
            secondary.append(dict(kind="secondary", sub="tap", pos=str(rng.choice(["pre", "hey_goo", "goo_gle"], p=[0.5, 0.3, 0.2]))))
    tense = float(np.clip(S["tension"] * rng.uniform(0.6, 1.4), 0.0, 1.0))
    return {"type": dtype, "events": ev + secondary, "profile": profile, "tense": round(tense, 3),
            "emotion": emotion or "n/a", "age": age or "n/a"}

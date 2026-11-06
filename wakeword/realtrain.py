"""Fine-tune the wake-word model with recordings of real people who stutter, and measure it honestly.

Why this is not "just add the clips to the training set":
  * 14 speakers x 1 clip cannot be split into train / test without wasting most of them, so the honest number comes
    from LEAVE-SPEAKERS-OUT cross-validation: every speaker is held out once and scored by a model that never heard it.
  * The real "Hey Google" clips have no Hey | Goo | gle boundaries and forced alignment fails on most of them, so they
    are used with a WEAK (clip-level) loss: each of Hey, Goo, gle must be heard somewhere in the clip (mean of the
    top-k frame log-posteriors of each unit).  Real Alexa / Siri clips are frame-labelled negatives.
  * Fine-tuning on ~12 clips would make the network forget everything else, so every step also trains on a replay
    sample of the original dataset, with batch-norm statistics frozen.
  * What the real clips fix most is CALIBRATION on real phone speech, so the verifier (decision layer) is refitted on
    synthetic validation clips AND the real training speakers together.

Two arms are compared in the cross-validation:  A = verifier only (frame model untouched),
B = frame-model fine-tuning + verifier.  The final model uses whichever arm is better in cross-validation.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import decoder, realpeople as rp
from .augment import augment_channel
from .compare import (FEATURE_SETS, GATES, MIN_MAX_LP, _fit_verifier_model, _oof_scores, sample_weights)
from .config import CLASSES, GLE, GOO, HEY, IGNORE, N_MELS, SR
from .decoder import FEATURE_NAMES
from .detector import Verifier, WakeWordDetector
from .features import pcen
from .model import load_sample, read_annotations, read_manifest

VERIFIER_KINDS = ("lr",)   # the MLP verifier saturated (every plausible phrase scored ~0.64): the threshold became a cliff
TOPK = 8                 # frames per unit used by the clip-level loss
REAL_NEG_WEIGHT = 5.0    # one real Alexa / Siri clip counts like this many synthetic negatives when choosing the threshold
P_DISFL = 0.35           # probability that a variant also gets simulated disfluencies (prolongation / repetition / block)
N_AUG = 14               # modulated copies of every real clip used in fine-tuning


# ----------------------------------------------------------------------------- data
class Real:
    """One real clip: waveform + label + speaker."""
    def __init__(self, row, x):
        self.id, self.spk, self.phrase = row["sample_id"], row["speaker"], row["phrase"]
        self.wake = int(row["is_wake_word"])
        self.cat = row["category"]
        self.x = x


def load_real(out_dir=rp.REAL_DIR):
    rows = rp.load_manifest(out_dir)
    return [Real(r, rp.load_clip(r, out_dir)) for r in rows]


def _aug_wave(x, rng):
    """Real-voice synthetic variants: a random voice modulation (rate incl. fast speech, pitch, whisper, shaky voice,
    intonation, band limit, reverb, small onset loss) followed by a random recording channel (gain, noise, ...)."""
    from .robust import disfluent, modulate
    y = disfluent(x, rng) if rng.random() < P_DISFL else x      # simulated stutter on the real voice
    y = modulate(x if y is x else y, rng)
    y, _ = augment_channel(y, rng, strength=1.0)
    return y.astype(np.float32)


def real_variants(items, n_aug, rng):
    """[(feat (T,N_MELS), frame labels or None, wake, item)] : the clean clip plus n_aug augmented copies."""
    out = []
    for it in items:
        lab0 = None if it.wake else rp.frame_labels_negative(it.x, it.phrase)
        for k in range(n_aug + 1):
            x = it.x if k == 0 else _aug_wave(it.x, rng)
            f = pcen(x)
            lab = None
            if not it.wake:
                lab = lab0 if k == 0 else rp.frame_labels_negative(x, it.phrase)
                n = min(len(f), len(lab))
                f, lab = f[:n], lab[:n].astype(np.int64).copy()
                lab[lab == IGNORE] = -100
            out.append((f, lab, it.wake, it))
    return out


def load_replay(replay_dir, max_train=None, seed=0, max_val=1000):
    """Replay sample of the original dataset: features + frame labels of its 'train' clips, rows of its 'val' clips."""
    from .torch_models import _load_clips
    rows, ann = read_manifest(replay_dir), read_annotations(replay_dir)
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    rng = np.random.default_rng(seed)
    if max_train and len(tr) > max_train:
        tr = [tr[i] for i in rng.permutation(len(tr))[:max_train]]
    if max_val and len(va) > max_val:
        va = [va[i] for i in sorted(rng.permutation(len(va))[:max_val])]
    f, l = _load_clips(replay_dir, tr, ann)
    return {"dir": replay_dir, "ann": ann, "train_f": f, "train_l": l, "train_rows": tr, "val_rows": va}



SPEECH_CATS = ("speech:",)
SPEECH_EXACT = ("tts:speech",)


def _is_talk(r):
    return r["is_wake_word"] == "0" and (r["category"].startswith(SPEECH_CATS) or r["category"] in SPEECH_EXACT)


def build_windows(replay, rows, n, rng, secs=(3.5, 6.0)):
    """Continuous-talk negatives: n windows of 3.5-6 s made of random ordinary-speech clips with short pauses and a
    random room / noise channel.  Returns [(waveform, frame labels)].  This is what a television or a conversation
    sounds like to a microphone: no pause long enough to reset the listener."""
    from .model import load_sample as _ls
    from .disfluency import segments_to_frame_labels  # noqa: F401  (labels come from load_sample)
    talk = [r for r in rows if _is_talk(r)]
    out = []
    if not talk:
        return out
    for _ in range(n):
        target = rng.uniform(*secs) * SR
        xs, ls, tot = [], [], 0
        while tot < target:
            r = talk[int(rng.integers(0, len(talk)))]
            x, lab = _ls(replay["dir"], r, replay["ann"])
            gap = int(rng.uniform(0.05, 0.5) * SR)
            xs += [x, np.zeros(gap, np.float32)]
            ls += [lab.astype(np.int64), np.zeros(max(gap // 160, 0), np.int64)]       # 0 = bg
            tot += len(x) + gap
        x = np.concatenate(xs)
        lab = np.concatenate(ls)
        y, _ = augment_channel(x, rng, strength=1.0)
        out.append((y.astype(np.float32), lab))
    return out


def make_crops(model, items, min_s=0.25):
    """Partial-phrase negatives from the real 'Hey Google' clips: 'Hey' alone (cut where Goo starts) and 'Google' alone
    (from where Goo starts).  The current model's own path gives the cut, which is only needed to within ~0.1 s."""
    out = []
    for it in items:
        if not it.wake:
            continue
        d = decoder.decode(model.predict_wave(it.x), None)
        if d["path"] is None or d["span"] is None:
            continue
        g = np.where(d["path"] == "G")[0]
        if len(g) == 0:
            continue
        cut = int(g.min()) * 160
        if cut > min_s * SR and len(it.x) - cut > min_s * SR:
            out.append((it, "hey_only", it.x[: cut + int(0.10 * SR)]))
            out.append((it, "google_only", it.x[max(cut - int(0.10 * SR), 0):]))
    return out


def _crop_variants(crops, n_aug, rng):
    out = []
    for it, kind, x in crops:
        for k in range(n_aug + 1):
            y = x if k == 0 else _aug_wave(x, rng)
            out.append((pcen(y), None, 0, it))
    return out


def _neg_clip_loss(lp):
    """Clip-level penalty for a clip that is NOT the wake word: the weakest of the three units' best frames must be
    low (so 'Hey' alone, 'Google' alone, 'Hey ... something else' cannot score as the phrase)."""
    mask = lp[:, 0] < np.log(0.5)
    if int(mask.sum()) < TOPK:
        mask = torch.ones(lp.shape[0], dtype=torch.bool)
    m = torch.stack([_topk_mean(lp, c, mask).exp() for c in (HEY, GOO, GLE)]).min()
    return -torch.log(1.0 - m * 0.999)



def aug_feat(f, rng, p=0.8):
    """Speaker / channel variation on a raw PCEN matrix (T, N_MELS): random vocal-tract-length-like warp of the mel axis
    (x0.88-1.12, Jaitly & Hinton 2013) and 1-2 short time masks (SpecAugment, Park et al. 2019).  Real voices are scarce
    in the training data (about 84 recordings against ~1,000 text-to-speech voices), so voice variety has to be created."""
    if rng.random() > p:
        return f
    a = float(rng.uniform(0.88, 1.12))
    pos = np.clip(np.arange(f.shape[1]) * a, 0, f.shape[1] - 1)
    lo = np.floor(pos).astype(int); hi = np.minimum(lo + 1, f.shape[1] - 1); w = (pos - lo).astype(np.float32)
    g = f[:, lo] * (1 - w) + f[:, hi] * w
    for _ in range(int(rng.integers(0, 3))):
        L = int(rng.integers(3, 11))
        if g.shape[0] > 3 * L:
            s0 = int(rng.integers(0, g.shape[0] - L)); g = g.copy(); g[s0:s0 + L] = g.mean(axis=0)
    return g.astype(np.float32)


def _batch_aug(feats, labs, idx, mean, std, rng, feat_aug=True):
    from .torch_models import _batch
    if not feat_aug:
        return _batch(feats, labs, idx, mean, std, freq_mask=True, rng=rng)
    f2 = [aug_feat(feats[i], rng) for i in idx]
    return _batch(f2, [labs[i] for i in idx], list(range(len(idx))), mean, std, freq_mask=True, rng=rng)


# ----------------------------------------------------------------------------- fine-tuning
def _topk_mean(lp, c, mask, k=TOPK):
    v = lp[:, c].masked_fill(~mask, -1e4)
    kk = min(k, int(mask.sum().item()))
    return torch.topk(v, kk).values.mean()


def finetune(model, replay, real_items, epochs=8, per_epoch=480, lr=3e-4, mil_weight=1.0, seed=0, log=print, n_windows=60, extra_negatives=False, feat_aug=True):
    """Return a fine-tuned COPY of `model` (a TorchFrameModel)."""
    import copy
    import torch.nn as nn
    from .torch_models import TorchFrameModel, _batch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = copy.deepcopy(model.net)
    mean, std = model.mean, model.std
    var = real_variants(real_items, N_AUG, rng)
    pos = [v for v in var if v[2]]
    neg = [v for v in var if not v[2]]
    # Extra clip-level / talk-window negatives were tried in v1.3 and LOWERED wake-word recall on the original test set
    # (AUC 0.947 -> 0.925), so they are off by default; the decision layer still learns from such negatives (decode_pool).
    crops = _crop_variants(make_crops(model, real_items), 3, rng) if extra_negatives else []
    wins = build_windows(replay, replay["train_rows"], n_windows, rng) if extra_negatives else []
    wf = [pcen(w) for w, _ in wins]
    wl = []
    for (w, lab), f in zip(wins, wf):
        n = min(len(f), len(lab)); lab = lab[:n].copy(); lab[lab == IGNORE] = -100
        wl.append(lab)
    wf = [f[:len(l)] for f, l in zip(wf, wl)]
    log(f"      fine-tune data: {len(pos)} positive / {len(neg)} negative real variants, {len(crops)} partial-phrase crops, {len(wf)} talk windows")
    # class weights as in the original training (fixed, mild)
    w = torch.tensor([0.6, 1.5, 1.5, 1.5, 0.9], dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w, ignore_index=-100)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    n_rep = len(replay["train_f"])
    t0 = time.time()
    for ep in range(epochs):
        net.eval()                                   # frozen batch-norm statistics + no dropout: stable with tiny batches
        tot, nb = 0.0, 0
        order = rng.permutation(n_rep)[:per_epoch]
        n_batches = max(1, len(order) // 12)
        for b in range(n_batches):
            idx = order[b * 12:(b + 1) * 12]
            X, Y = _batch_aug(replay["train_f"], replay["train_l"], idx, mean, std, rng, feat_aug)
            opt.zero_grad()
            loss = lossf(net(X).reshape(-1, len(CLASSES)), Y.reshape(-1))
            if neg:                                   # real negatives (frame labelled)
                sel = [neg[int(i)] for i in rng.integers(0, len(neg), 2)]
                f_l = [s[0] for s in sel]
                l_l = [s[1] for s in sel]
                Xn, Yn = _batch_aug(f_l, l_l, list(range(len(sel))), mean, std, rng, feat_aug)
                loss = loss + lossf(net(Xn).reshape(-1, len(CLASSES)), Yn.reshape(-1))
                for s_ in (sel if extra_negatives else []):    # ... and the clip must not contain all three units
                    xn = torch.from_numpy(((s_[0] - mean) / std).astype(np.float32))[None]
                    loss = loss + 0.5 * _neg_clip_loss(torch.log_softmax(net(xn), dim=-1)[0])
            if crops:
                for s_ in [crops[int(i)] for i in rng.integers(0, len(crops), 2)]:
                    xn = torch.from_numpy(((s_[0] - mean) / std).astype(np.float32))[None]
                    loss = loss + 0.7 * _neg_clip_loss(torch.log_softmax(net(xn), dim=-1)[0])
            if wf:
                wi = [int(i) for i in rng.integers(0, len(wf), 2)]
                Xw, Yw = _batch(wf, wl, wi, mean, std)
                loss = loss + 0.7 * lossf(net(Xw).reshape(-1, len(CLASSES)), Yw.reshape(-1))
            if pos and mil_weight > 0:                # real positives: weak clip-level loss
                sel = [pos[int(i)] for i in rng.integers(0, len(pos), 3)]
                m = 0.0
                for f, _, _, _ in sel:
                    xp = torch.from_numpy(((aug_feat(f, rng) if feat_aug else f) - mean) / std).float()[None]
                    lp = torch.log_softmax(net(xp), dim=-1)[0]
                    mask = torch.ones(lp.shape[0], dtype=torch.bool)
                    voiced = lp[:, 0] < np.log(0.5)    # frames the model does not call background
                    mask = voiced if int(voiced.sum()) >= TOPK else mask
                    m = m - sum(_topk_mean(lp, c, mask) for c in (HEY, GOO, GLE)) / 3.0
                loss = loss + mil_weight * m / len(sel) * 0.5
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        sched.step()
        log(f"      fine-tune epoch {ep + 1}/{epochs}  loss={tot / max(nb, 1):.3f}  ({time.time() - t0:.0f}s)")
    net.eval()
    return TorchFrameModel(net, mean, std, model.arch)


# ----------------------------------------------------------------------------- verifier on synthetic val + real speakers
def decode_pool(model, replay, real_items, n_aug_real=2, seed=0, n_windows=150, hard_negs=None):
    """Decode (no unit gate) the synthetic validation clips and the real clips (clean + n_aug_real augmented copies).
    Returns dict of arrays: X, y, grp, lab, cat, kind, plus per-clip bookkeeping for clips with no path."""
    rng = np.random.default_rng(seed)
    X, y, grp, lab, cat, kind, real_flag = [], [], [], [], [], [], []
    n_neg_synth = 0
    n_pos = {"fluent": 0, "disfluent": 0}
    for r in replay["val_rows"]:
        x, _ = load_sample(replay["dir"], r, replay["ann"])
        d = decoder.decode(model.predict_wave(x), None)
        if r["is_wake_word"] == "1":
            n_pos[r["label"]] = n_pos.get(r["label"], 0) + 1
        else:
            n_neg_synth += 1
        if d["features"] is None:
            continue
        X.append(d["features"]); y.append(int(r["is_wake_word"])); grp.append(r["group"]); lab.append(r["label"])
        cat.append(r["category"]); kind.append(r.get("speaker_kind", "")); real_flag.append(0)
    for w, _ in build_windows(replay, replay["val_rows"], n_windows, rng):
        n_neg_synth += 1
        d = decoder.decode(model.predict_wave(w), None)
        if d["features"] is None:
            continue
        X.append(d["features"]); y.append(0); grp.append("WIN"); lab.append("negative")
        cat.append("hard:window"); kind.append("window"); real_flag.append(0)
    for w in (hard_negs or []):                      # mined false triggers from large negative audio
        n_neg_synth += 1
        d = decoder.decode(model.predict_wave(w), None)
        if d["features"] is None:
            continue
        X.append(d["features"]); y.append(0); grp.append("MINED"); lab.append("negative")
        cat.append("hard:mined"); kind.append("mined"); real_flag.append(0)
    for it, ck, cx in make_crops(model, real_items):
        for k in range(2):
            d = decoder.decode(model.predict_wave(cx if k == 0 else _aug_wave(cx, rng)), None)
            if d["features"] is None:
                continue
            X.append(d["features"]); y.append(0); grp.append(f"REAL_{it.spk}"); lab.append("real_neg")
            cat.append("tts:partial"); kind.append("real_stutter"); real_flag.append(1)
    for it in real_items:
        for k in range(n_aug_real + 1):
            x = it.x if k == 0 else _aug_wave(it.x, rng)
            d = decoder.decode(model.predict_wave(x), None)
            if d["features"] is None:
                continue
            X.append(d["features"]); y.append(it.wake); grp.append(f"REAL_{it.spk}")
            lab.append("real_stutter" if it.wake else "real_neg")
            cat.append("real_wake" if it.wake else "tts:other_wake"); kind.append("real_stutter"); real_flag.append(1)
    return {"X": np.array(X), "y": np.array(y), "grp": np.array(grp), "lab": np.array(lab), "cat": np.array(cat),
            "kind": np.array(kind), "real": np.array(real_flag), "n_neg_synth": n_neg_synth, "n_pos": n_pos,
            "n_real_items": len(real_items)}


def _wthr(s_neg, w_neg, far):
    """Lowest threshold whose weighted false-accept mass stays within far x the total negative mass."""
    o = np.argsort(-s_neg)
    s, w = s_neg[o], w_neg[o]
    budget = far * w.sum()
    c = np.cumsum(w)
    k = int(np.searchsorted(c, budget, side="right"))
    return float(s[k]) if k < len(s) else 0.0


def fit_verifier(pool, target_far=0.01, log=print, verbose=False):
    """Choose feature set x unit gate x verifier by out-of-fold worst-group TPR (fluent, disfluent, REAL speakers) and
    fit it on everything.  Real negatives are weighted REAL_NEG_WEIGHT x in the threshold."""
    from sklearn.metrics import roc_auc_score
    X0, y0, g0, lab0, cat0, kind0, real0 = (pool[k] for k in ("X", "y", "grp", "lab", "cat", "kind", "real"))
    n_neg_all = pool["n_neg_synth"] + int(sum(1 for i in range(len(y0)) if real0[i] and y0[i] == 0))
    best = None
    for fset, names in FEATURE_SETS.items():
        idx = [FEATURE_NAMES.index(n) for n in names]
        for gate in GATES:
            m = X0[:, MIN_MAX_LP] >= np.log(gate)
            X, y, g, lab, cat, kind, real = X0[m], y0[m], g0[m], lab0[m], cat0[m], kind0[m], real0[m]
            if y.sum() < 10 or (y == 0).sum() < 10:
                continue
            w = sample_weights(y, np.where(real == 1, "disfluent", lab), cat, np.where(real == 1, "real_recording", kind))
            for vk in VERIFIER_KINDS:
                oof = _oof_scores(X, y, g, vk, w, idx)
                wn = np.where(real[y == 0] == 1, REAL_NEG_WEIGHT, 1.0)
                thr = _wthr(oof[y == 0], wn, target_far)
                hit = (oof > thr) & (y == 1)
                tpr = {"fluent": hit[(lab == "fluent")].sum() / max(pool["n_pos"].get("fluent", 1), 1),
                       "disfluent": hit[(lab == "disfluent")].sum() / max(pool["n_pos"].get("disfluent", 1), 1)}
                nr = int(((real == 1) & (y == 1)).sum())
                tpr["real"] = float(hit[(real == 1)].sum() / nr) if nr else 1.0
                worst = min(tpr.values())
                auc = float(roc_auc_score(y, oof))
                if verbose:
                    log(f"     {fset:14s} gate {gate:.2f} {vk:3s}: fluent={tpr['fluent']:.3f} disf={tpr['disfluent']:.3f} real={tpr['real']:.3f} AUC={auc:.3f}")
                if best is None or (worst, auc) > (best["worst"], best["auc"]):
                    best = {"vk": vk, "gate": gate, "fset": fset, "idx": idx, "worst": float(worst), "auc": auc, "thr": thr,
                            "tpr": {k: round(float(v), 4) for k, v in tpr.items()}, "data": (X, y, w)}
    X, y, w = best["data"]
    ver, _ = _fit_verifier_model(X, y, best["vk"], w=w, idx=best["idx"])
    ver.threshold, ver.unit_gate = float(best["thr"]), float(best["gate"])
    info = {k: best[k] for k in ("vk", "gate", "fset", "worst", "auc", "tpr")}
    info["threshold"] = round(ver.threshold, 4)
    return ver, info


# ----------------------------------------------------------------------------- scoring held-out speakers
def score_items(det, items):
    out = []
    for it in items:
        r = det.analyze(it.x, repair=False, with_curves=False)
        out.append({"id": it.id, "speaker": it.spk, "phrase": it.phrase, "wake": it.wake, "category": it.cat,
                    "score": float(r["score"]), "hit": bool(r["detected"]), "heard": r.get("heard_text", ""),
                    "reason": r.get("reason", "")})
    return out


def battery_scores(det, items, seed=0):
    """{transform: [(id, wake, score, threshold)]} : every clip under every transform of robust.BATTERY."""
    from .robust import BATTERY
    out = {}
    for name, f in BATTERY.items():
        rng = np.random.default_rng(seed)
        rows = []
        for it in items:
            y = f(it.x, rng)
            rows.append((it.id, it.wake, float(det.analyze(y, repair=False, with_curves=False)["score"]), float(det.threshold)))
        out[name] = rows
    return out


def battery_table(batt) -> str:
    """Markdown: recall of 'Hey Google' clips / false accepts of other phrases per transform and arm."""
    arms = list(batt)
    names = list(next(iter(batt.values())))
    L = ["| transform | " + " | ".join(f"{a}: Hey-Google accepted / other accepted" for a in arms) + " |", "|---|" + "---|" * len(arms)]
    for n in names:
        cells = []
        for a in arms:
            r = batt[a][n]
            p = [x for x in r if x[1]]; q = [x for x in r if not x[1]]
            cells.append(f"{sum(x[2] >= x[3] for x in p)}/{len(p)} , {sum(x[2] >= x[3] for x in q)}/{len(q)}")
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def _folds(speakers, k, seed=0):
    sp = sorted(speakers)
    rng = np.random.default_rng(seed)
    rng.shuffle(sp)
    return [sorted(sp[i::k]) for i in range(k)]


def cross_validate(model_dir, replay_dir, real_dir=rp.REAL_DIR, k=7, arms=("A", "B"), epochs=8, per_epoch=480,
                   max_replay=1600, target_far=0.01, seed=0, out=None, log=print, tuner=None, hard_negs=None):
    """Leave-speakers-out CV.  Returns {arm: summary}; also summarises the unchanged model as 'baseline'."""
    base = WakeWordDetector.load(model_dir)
    items = load_real(real_dir)
    speakers = sorted({it.spk for it in items})
    replay = load_replay(replay_dir, max_train=max_replay, seed=seed)
    folds = _folds(speakers, min(k, len(speakers)), seed)
    log(f"[cv] {len(items)} clips, {len(speakers)} speakers, {len(folds)} folds; replay {len(replay['train_f'])} train / "
        f"{len(replay['val_rows'])} val clips")
    res = {"baseline": [], **{a: [] for a in arms}}
    info = {a: [] for a in arms}
    res["baseline"] = score_items(base, items)
    batt = {"baseline": battery_scores(base, items, seed)}
    for a in arms:
        batt[a] = {}
    for fi, held in enumerate(folds):
        tr = [it for it in items if it.spk not in held]
        te = [it for it in items if it.spk in held]
        log(f"[cv] fold {fi + 1}/{len(folds)}  held-out speakers {held}  (train {len(tr)} clips, test {len(te)})")
        for arm in arms:
            t0 = time.time()
            model = base.model if arm == "A" else (tuner or finetune)(base.model, replay, tr, epochs, per_epoch, seed=seed + fi, log=lambda *_: None)
            pool = decode_pool(model, replay, tr, seed=seed + fi, hard_negs=hard_negs)
            ver, vi = fit_verifier(pool, target_far)
            det = WakeWordDetector(model, ver)
            sc = score_items(det, te)
            for s in sc:
                s["threshold"] = ver.threshold
            res[arm] += sc
            for k_, rows_ in battery_scores(det, te, seed + fi).items():
                batt[arm].setdefault(k_, []).extend(rows_)
            info[arm].append({"held_out": held, **vi})
            log(f"      arm {arm}: {vi['fset']}/{vi['vk']} gate {vi['gate']} thr {vi['threshold']}  ({time.time() - t0:.0f}s)  "
                f"held-out hits {sum(s['hit'] for s in sc if s['wake'])}/{sum(s['wake'] for s in sc)}  "
                f"false accepts {sum(s['hit'] for s in sc if not s['wake'])}/{sum(1 for s in sc if not s['wake'])}")
    summ = {}
    for name, recs in res.items():
        thr = base.threshold if name == "baseline" else float(np.mean([s["threshold"] for s in recs]))
        if name == "baseline":
            for s in recs:
                s["threshold"] = thr
        summ[name] = rp.summarise(recs, thr, name)
        summ[name]["clips"] = recs
        # pooled AUC mixes thresholds of different folds, so also give the threshold-free per-clip scores
    report = {"folds": folds, "k": len(folds), "epochs": epochs, "summary": summ, "verifier_choices": info, "battery": batt}
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (Path(out) / "cv_report.md").write_text(cv_markdown(report), encoding="utf-8")
    log(cv_markdown(report).encode("ascii", "replace").decode("ascii"))
    return report


def cv_markdown(rep) -> str:
    L = ["# Leave-speakers-out cross-validation on real people who stutter", "",
         f"{rep['k']} folds; every speaker is scored by a model that never heard that speaker (neither the frame model nor "
         "the verifier / threshold). Arms: baseline = the installed model unchanged, A = verifier refitted with the real "
         "training speakers, B = A + frame-model fine-tuning.", "",
         "| arm | 'Hey Google' accepted (95% CI) | other phrases accepted (95% CI) | AUC |", "|---|---|---|---|"]
    for name, s in rep["summary"].items():
        L.append(f"| {name} | {s['detected']}/{s['n_wake']} ({s['recall_95ci'][0]}-{s['recall_95ci'][1]}) | "
                 f"{s['false_accepts']}/{s['n_negative']} ({s['FAR_95ci'][0]}-{s['FAR_95ci'][1]}) | {s['AUC']} |")
    L += ["", "With 14 'Hey Google' clips every interval above is wide: a difference of one or two clips is NOT significant."]
    for name, s in rep["summary"].items():
        L += ["", f"## {name}: per-clip scores", "", "| clip | speaker | score | threshold | accepted |", "|---|---|---|---|---|"]
        for r in sorted(s["clips"], key=lambda r: (r["phrase"], str(r["speaker"]))):
            L.append(f"| {r['id']} | {r['speaker']} | {r['score']:.3f} | {r.get('threshold', s['threshold']):.3f} | {'yes' if r['hit'] else 'no'} |")
    if rep.get("battery"):
        L += ["", "## Robustness battery on held-out speakers (clips transformed, scored by a model that never heard the speaker)", "",
              battery_table(rep["battery"])]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- final model
def final_train(model_dir, replay_dir, real_dir=rp.REAL_DIR, arm="B", epochs=8, per_epoch=480, max_replay=1600,
                target_far=0.01, seed=0, install=True, log=print, extra_negatives=False, tuner=None, hard_negs=None):
    """Fine-tune on ALL speakers (arm B) or only refit the verifier (arm A), then install with a backup."""
    model_dir = Path(model_dir)
    base = WakeWordDetector.load(model_dir)
    items = load_real(real_dir)
    replay = load_replay(replay_dir, max_train=max_replay, seed=seed)
    model = base.model if arm == "A" else (tuner(base.model, replay, items, epochs, per_epoch, seed=seed, log=log) if tuner else finetune(base.model, replay, items, epochs, per_epoch, seed=seed, log=log, extra_negatives=extra_negatives))
    pool = decode_pool(model, replay, items, seed=seed, hard_negs=hard_negs)
    ver, vi = fit_verifier(pool, target_far, log=log, verbose=True)
    det = WakeWordDetector(model, ver)
    log(f"[final] arm {arm}: verifier {vi}")
    if install:
        bk = model_dir / "before_real_people"
        bk.mkdir(exist_ok=True)
        for f in ("frame_model.pt", "frame_model.npz", "verifier.json", "meta.json"):
            if (model_dir / f).exists() and not (bk / f).exists():
                shutil.copy(model_dir / f, bk / f)
        if arm == "B":
            model.save(model_dir / "frame_model.pt")
        (model_dir / "verifier.json").write_text(json.dumps(ver.to_json()), encoding="utf-8")
        meta_p = model_dir / "meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        if arm == "B":
            meta.update({"frame_type": getattr(model, "arch", "crnn"), "frame_file": "frame_model.pt", "name": getattr(model, "arch", "crnn")})
        meta.update({"verifier": vi["vk"], "unit_gate": vi["gate"], "verifier_features": vi["fset"],
                     "real_people": {"arm": arm, "clips": len(items), "speakers": len({i.spk for i in items})}})
        meta_p.write_text(json.dumps(meta), encoding="utf-8")
        log(f"[final] installed in {model_dir}  (previous files kept in {bk})")
    return det, vi


import torch  # noqa: E402  (bottom import: the helpers above use it at call time)

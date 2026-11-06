"""Train several candidate models, compare them on held-out data and keep the best one.

Every candidate = frame model (MLP / CNN / CRNN) + disfluency-tolerant decoder + verifier.
Ranking uses ONLY the validation split: the verifier scores are out-of-fold (grouped 5-fold by
recording / speaker), so the number is not inflated by fitting. The test split is used once,
afterwards, for the final report. Candidates that cannot run on this machine (e.g. PyTorch not
installed) are skipped with a message - they never stop the run.
"""
from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from . import evaluate as evaluate_mod
from .config import DATA_DIR, MODEL_DIR
from .decoder import FEATURE_NAMES
from .detector import Verifier, WakeWordDetector
from .model import FrameModel, read_annotations, read_manifest, train_frame_model
from .train import decode_split

CANDIDATES = {
    "mlp_small": {"type": "mlp", "hidden": (128, 64), "desc": "scikit-learn MLP 128-64 over stacked PCEN context"},
    "mlp_large": {"type": "mlp", "hidden": (384, 192), "desc": "scikit-learn MLP 384-192 over stacked PCEN context"},
    "cnn": {"type": "cnn", "desc": "PyTorch dilated temporal CNN (needs torch)"},
    "crnn": {"type": "crnn", "desc": "PyTorch CNN + bidirectional GRU (needs torch)"},
}
FILES = {"mlp": "frame_model.npz", "cnn": "frame_model.pt", "crnn": "frame_model.pt"}


# The verifier reads only LENGTH-INDEPENDENT evidence: how well each unit matched, not how long the phrase was or
# how often a unit was repeated.  The v1.1 verifier also read span length, frame counts and restart counts and learnt
# positive weights for them (more stuttering = higher score), so a long stuttered 'hey' scored above a short fluent
# 'Hey Google'.  Disfluency must be TOLERATED by the decoder, never REWARDED by the verifier.
VERIFIER_FEATURES = ["deficit_per_frame", "mean_lp_hey", "mean_lp_goo", "mean_lp_gle", "min_mean_lp",
                     "frac_other_in_span", "max_lp_hey", "max_lp_goo", "max_lp_gle", "min_max_lp"]
VERIFIER_IDX = [FEATURE_NAMES.index(n) for n in VERIFIER_FEATURES]
# v1.2d: the per-frame deficit includes the grammar's restart (-1.2 each) and filler penalties, and the per-unit means
# are diluted by short stutter fragments, so they mark down a stuttered 'Hey Google' that is heard perfectly.
# The structure-free set reads only the unit frames themselves; validation decides which set is used.
FEATURE_SETS = {
    "v1": VERIFIER_FEATURES,
    "structure_free": ["unit_deficit_pf", "top5_hey", "top5_goo", "top5_gle", "min_top5", "gap_quality",
                       "max_lp_hey", "max_lp_goo", "max_lp_gle", "min_max_lp", "frac_other_in_span"],
}
# negatives that sound like the keyword count more: they are what a deployed detector actually gets wrong
HARD_NEG_WEIGHT = 3.0
HARD_NEG_PREFIXES = ("hard:", "tts:near_miss", "tts:hard_google", "tts:partial", "tts:other_wake")


def sample_weights(y, labels, cats, kinds=None):
    """Fluent and disfluent positives get the SAME total weight (equal to the negatives' total); hard negatives x3;
    real-recording positives x3 inside their group (a live microphone is a real recording, not a TTS voice)."""
    w = np.ones(len(y), dtype=np.float64)
    labels, cats = np.asarray(labels), np.asarray(cats)
    neg = y == 0
    hard = neg & np.array([c.startswith(HARD_NEG_PREFIXES) for c in cats])
    w[hard] = HARD_NEG_WEIGHT
    w[neg] *= len(y) / 2 / max(w[neg].sum(), 1e-9)
    real = np.zeros(len(y), bool) if kinds is None else (np.asarray(kinds) == "real_recording")
    for lab in ("fluent", "disfluent"):
        m = (y == 1) & (labels == lab)
        if m.any():
            raw = np.where(real[m], 3.0, 1.0)
            w[m] = raw * (len(y) / 4 / raw.sum())
    return w


def _fit_verifier_model(X, y, kind: str, seed: int = 0, w=None, idx=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    idx = VERIFIER_IDX if idx is None else list(idx)
    X = X[:, idx]
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-3
    Xn = (X - mean) / std
    if kind == "lr":
        m = LogisticRegression(C=0.5, max_iter=3000).fit(Xn, y, sample_weight=w)
        return Verifier(mean, std, [(m.coef_.T, m.intercept_)], 0.5, idx), m
    rng = np.random.default_rng(seed)
    rs = rng.choice(len(y), size=2 * len(y), p=w / w.sum())           # weighted resampling (MLP has no sample_weight)
    m = MLPClassifier((32, 16), alpha=0.1, max_iter=600, early_stopping=True, validation_fraction=0.15,
                      random_state=seed).fit(Xn[rs], y[rs])
    layers = list(zip(m.coefs_, m.intercepts_))
    return Verifier(mean, std, layers, 0.5, idx), m


def _oof_scores(X, y, groups, kind, w=None, idx=None):
    from sklearn.model_selection import GroupKFold
    s = np.zeros(len(y))
    w = np.ones(len(y)) if w is None else w
    n_splits = int(min(5, len(set(groups))))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        ver, _ = _fit_verifier_model(X[tr], y[tr], kind, w=w[tr], idx=idx)
        s[te] = [ver.score(f) for f in X[te]]
    return s


def _thr_at_far(s, y, far, n_neg_all):
    """Lowest threshold whose false accepts stay within far x ALL validation negatives
    (negatives with no keyword path are already rejected and count in the denominator)."""
    neg = np.sort(s[y == 0])[::-1]
    k = int(far * n_neg_all)
    return float(neg[k]) if k < len(neg) else 0.0


def _tpr_at_far(s, y, far):
    neg = np.sort(s[y == 0])[::-1]
    t = neg[int(far * len(neg))] if len(neg) else 1.0
    return float((s[y == 1] > t).mean()), float(t)


GROUPS = ("fluent", "disfluent", "real")
GATES = (0.10, 0.15, 0.20, 0.30)              # candidate unit gates, chosen on validation data
MIN_MAX_LP = FEATURE_NAMES.index("min_max_lp")


def _val_decode(model, data_dir, rows, ann, log):
    """Decode WITHOUT the unit gate (the gate is applied afterwards for every candidate value)."""
    dec = decode_split(model, data_dir, rows, ann, log, gate=None)
    X, y, g, lab, cat, kind = [], [], [], [], [], []
    for r, d in dec:
        if d["features"] is None:                                   # no Hey->Goo->gle path at all
            continue
        X.append(d["features"]); y.append(int(r["is_wake_word"])); g.append(r["group"])
        lab.append(r["label"]); cat.append(r["category"]); kind.append(r.get("speaker_kind", ""))
    n_pos = {k: sum(1 for r, _ in dec if r["label"] == k) for k in ("fluent", "disfluent")}
    n_pos["real"] = sum(1 for r, _ in dec if r["is_wake_word"] == "1" and r.get("speaker_kind") == "real_recording")
    return (np.array(X), np.array(y), np.array(g), np.array(lab), np.array(cat), np.array(kind), n_pos, len(dec))


def _group_tpr(s, y, lab, kind, thr, n_pos):
    """TPR of fluent / disfluent / real-recording wake words at threshold thr; clips without a path count as missed."""
    hit = (s > thr) & (y == 1)
    out = {k: float((hit & (lab == k)).sum() / max(n_pos[k], 1)) for k in ("fluent", "disfluent")}
    out["real"] = float((hit & (kind == "real_recording")).sum() / max(n_pos["real"], 1)) if n_pos["real"] else 1.0
    return out


MIN_CLIPS_FOR_REAL_RUN = 2500


def select_verifier(model, data_dir, val, ann, target_far=0.01, log=print):
    """Decode the validation clips once (without the unit gate) and choose, out-of-fold, the feature set x unit gate x
    verifier with the highest WORST-GROUP TPR (fluent / disfluent / real voices) at the target false accept rate.
    Returns (fitted Verifier, best-choice dict, TPR-by-FAR curve)."""
    from sklearn.metrics import roc_auc_score
    X0, y0, g0, lab0, cat0, kind0, n_pos, n_val = _val_decode(model, data_dir, val, ann, log)
    n_neg_all = sum(1 for r in val if r["is_wake_word"] != "1")
    best = None
    for fset, names in FEATURE_SETS.items():
        idx = [FEATURE_NAMES.index(n) for n in names]
        for gate in GATES:
            m = X0[:, MIN_MAX_LP] >= np.log(gate)
            X, y, g, lab, cat, kind = X0[m], y0[m], g0[m], lab0[m], cat0[m], kind0[m]
            if y.sum() < 10 or (y == 0).sum() < 10:
                continue
            w = sample_weights(y, lab, cat, kind)
            for vk in ("lr", "mlp"):
                oof = _oof_scores(X, y, g, vk, w, idx)
                thr = _thr_at_far(oof, y, target_far, n_neg_all)
                tg = _group_tpr(oof, y, lab, kind, thr, n_pos)
                worst = min(tg.values())
                auc = float(roc_auc_score(y, oof))
                log(f"     {fset:14s} gate {gate:.2f} {vk:3s}: TPR fluent={tg['fluent']:.3f} disfluent={tg['disfluent']:.3f} "
                    f"real={tg['real']:.3f}  (worst {worst:.3f})  AUC={auc:.3f}")
                if best is None or (worst, auc) > (best["val_worst_TPR"], best["val_AUC"]):
                    best = {"verifier": vk, "gate": gate, "fset": fset, "idx": idx, "val_worst_TPR": round(worst, 4),
                            "val_AUC": round(auc, 4), **{f"val_TPR_{k}": round(v, 4) for k, v in tg.items()},
                            "oof_threshold": thr, "oof": oof, "data": (X, y, lab, kind, w)}
    if best is None:
        raise RuntimeError("no gate left enough validation clips")
    X, y, lab, kind, w = best["data"]
    curve = {}
    for f in (0.005, 0.01, 0.02, 0.05):
        tg = _group_tpr(best["oof"], y, lab, kind, _thr_at_far(best["oof"], y, f, n_neg_all), n_pos)
        curve[f"{f:.1%}"] = {k: round(v, 4) for k, v in tg.items()}
    ver, _ = _fit_verifier_model(X, y, best["verifier"], w=w, idx=best["idx"])
    ver.threshold = float(best["oof_threshold"])
    ver.unit_gate = float(best["gate"])
    log(f"  -> chosen verifier: {best['verifier']} on '{best['fset']}' features, gate {best['gate']}, threshold "
        f"{ver.threshold:.4f}; val TPR fluent={best['val_TPR_fluent']} disfluent={best['val_TPR_disfluent']} "
        f"real={best['val_TPR_real']}")
    return ver, best, curve


def refit_verifier(model_dir=MODEL_DIR, data_dir=DATA_DIR, target_far: float = 0.01, log=print):
    """Re-choose and refit ONLY the verifier (feature set, unit gate, threshold) on the installed frame model.
    The frame model is unchanged, so this takes minutes instead of a full training run."""
    model_dir, data_dir = Path(model_dir), Path(data_dir)
    det = WakeWordDetector.load(model_dir)
    rows, ann = read_manifest(data_dir), read_annotations(data_dir)
    val = [r for r in rows if r["split"] == "val"]
    log(f"refitting the verifier of the model in {model_dir} on {len(val)} validation clips ...")
    ver, best, curve = select_verifier(det.model, data_dir, val, ann, target_far, log)
    old = model_dir / "verifier.json"
    if old.exists():
        (model_dir / "verifier_before_refit.json").write_text(old.read_text(encoding="utf-8"), encoding="utf-8")
    old.write_text(json.dumps(ver.to_json()), encoding="utf-8")
    meta_p = model_dir / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    meta.update({"verifier": best["verifier"], "unit_gate": best["gate"], "verifier_features": best["fset"],
                 "target_far": target_far})
    meta_p.write_text(json.dumps(meta), encoding="utf-8")
    log("     val TPR (fluent / disfluent / real) at other false accept rates: "
        + "  ".join(f"FAR{k}={v['fluent']}/{v['disfluent']}/{v['real']}" for k, v in curve.items()))
    log(f"saved {old}  (previous verifier kept as verifier_before_refit.json)")


def run_comparison(data_dir=DATA_DIR, model_dir=MODEL_DIR, names=None, epochs_mlp: int = 12, epochs_torch: int = 30,
                   target_far: float = 0.01, log=print, evaluate_test: bool = True, torch_max_clips=None, force: bool = False,
                   init_from=None, lr=None):
    t00 = time.time()
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    rows, ann = read_manifest(data_dir), read_annotations(data_dir)
    n_train = sum(1 for r in rows if r["split"] == "train")
    if n_train < MIN_CLIPS_FOR_REAL_RUN * 0.6 and not force:
        raise SystemExit(
            f"\nSTOP: the dataset in '{data_dir}' has only {len(rows)} clips ({n_train} for training) - that is the small DEMO subset "
            "shipped with the zip.\nTraining on it would give meaningless numbers and would overwrite your good model.\n"
            "Build the full dataset first:   python run.py build-data   (or the one-command route:  python run.py all)\n"
            "(add --force to compare anyway.)")
    val = [r for r in rows if r["split"] == "val"]
    try:
        from .torch_models import HAVE_TORCH, train_torch
    except Exception:
        HAVE_TORCH, train_torch = False, None
    names = names or list(CANDIDATES)
    results, built = [], {}
    for name in names:
        spec = CANDIDATES[name]
        cdir = model_dir / "candidates" / name
        cdir.mkdir(parents=True, exist_ok=True)
        log(f"\n===== candidate: {name}  ({spec['desc']}) =====")
        if spec["type"] in ("cnn", "crnn") and not HAVE_TORCH:
            log("  skipped: PyTorch is not installed (see README step 'Optional: PyTorch models').")
            results.append({"name": name, "status": "skipped (no torch)", "desc": spec["desc"]})
            continue
        t0 = time.time()
        try:
            fpath = cdir / FILES[spec["type"]]
            if spec["type"] == "mlp":
                model, hist = train_frame_model(data_dir, fpath, epochs=epochs_mlp, hidden=spec["hidden"], log=log)
            else:
                init = None
                if init_from:                       # fine-tune from a trained model of the same architecture
                    for cand in (Path(init_from) / "candidates" / name / "frame_model.pt", Path(init_from) / "frame_model.pt"):
                        if cand.exists():
                            init = cand
                            break
                model, hist = train_torch(spec["type"], data_dir, fpath, epochs=epochs_torch, log=log, max_train_clips=torch_max_clips,
                                          init_from=init, lr=lr)
            ver, best, curve = select_verifier(model, data_dir, val, ann, target_far, log)
            (cdir / "verifier.json").write_text(json.dumps(ver.to_json()), encoding="utf-8")
            meta = {"name": name, "frame_type": spec["type"], "frame_file": FILES[spec["type"]],
                    "verifier": best["verifier"], "target_far": target_far, "unit_gate": best["gate"],
                    "verifier_features": best["fset"]}
            (cdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            res = {"name": name, "status": "ok", "desc": spec["desc"], "verifier": f"{best['verifier']}/{best['fset']}",
                   "val_TPR_at_FAR": best["val_worst_TPR"], "val_TPR_fluent": best["val_TPR_fluent"],
                   "val_TPR_disfluent": best["val_TPR_disfluent"], "val_TPR_real": best["val_TPR_real"], "unit_gate": best["gate"],
                   "val_AUC": best["val_AUC"], "val_target_FAR": target_far,
                   "val_TPR_by_FAR": curve, "threshold": round(ver.threshold, 4), "train_seconds": round(time.time() - t0), "frame_hist": hist}
            results.append(res)
            built[name] = cdir
            log(f"  -> val at FAR {target_far:.1%}: TPR fluent={res['val_TPR_fluent']}  disfluent={res['val_TPR_disfluent']}  "
                f"real voices={res['val_TPR_real']}  (worst={res['val_TPR_at_FAR']})  AUC={res['val_AUC']}  verifier={res['verifier']}  "
                f"gate={res['unit_gate']}  threshold={res['threshold']}")
            log("     val TPR (fluent / disfluent / real) at other false accept rates (out-of-fold): "
                + "  ".join(f"FAR{k}={v['fluent']}/{v['disfluent']}/{v['real']}" for k, v in curve.items()))
        except Exception as exc:                                    # one broken candidate must not stop the rest
            log(f"  candidate {name} FAILED and was skipped: {exc}")
            log(traceback.format_exc(limit=3))
            results.append({"name": name, "status": f"failed: {exc}", "desc": spec["desc"]})
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        raise SystemExit("No candidate could be trained. Check the log above.")
    ok.sort(key=lambda r: (r["val_TPR_at_FAR"], r["val_AUC"]), reverse=True)
    winner = ok[0]["name"]
    log(f"\n>>> selected model: {winner}")
    best_dir = built[winner]
    # evaluate the winner FROM ITS CANDIDATE FOLDER first; the live model is replaced only after everything succeeded
    det = WakeWordDetector.load(best_dir)
    test_metrics = None
    if evaluate_test:
        evaluate_mod.evaluate(det, data_dir, "test", best_dir, log=lambda *_: None)
        for r in results:                                            # informational test numbers for every candidate
            if r["status"] == "ok":
                d = WakeWordDetector.load(built[r["name"]])
                m = evaluate_mod.evaluate(d, data_dir, "test", built[r["name"]], log=lambda *_: None)
                r.update({"test_FRR": m["FRR_overall"], "test_FAR": m["FAR_overall"], "test_AUC": m["AUC"],
                          "test_false_accepts_per_hour": m["false_accepts_per_hour"]})
    out = {"selected": winner, "selection_rule": f"highest WORST-GROUP validation TPR (min of fluent, disfluent, real-voice) at FAR<={target_far:.1%} (out-of-fold, unit gate and verifier chosen jointly), ties by AUC",
           "candidates": results, "total_minutes": round((time.time() - t00) / 60, 1)}
    (model_dir / "comparison.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    (model_dir / "comparison.md").write_text(_markdown(out), encoding="utf-8")
    # install the winner (keep the previous model in models/previous/)
    prev = model_dir / "previous"
    if (model_dir / "meta.json").exists():
        prev.mkdir(exist_ok=True)
        for f in ("frame_model.npz", "frame_model.pt", "verifier.json", "meta.json"):
            if (model_dir / f).exists():
                shutil.copy(model_dir / f, prev / f)
    for f in ("frame_model.npz", "frame_model.pt"):                  # drop a stale file of another type
        if (model_dir / f).exists() and not (best_dir / f).exists():
            (model_dir / f).unlink()
    for f in ("frame_model.npz", "frame_model.pt", "verifier.json", "meta.json", "metrics.json", "metrics_test.json", "report_test.md"):
        if (best_dir / f).exists():
            shutil.copy(best_dir / f, model_dir / f)
    det = WakeWordDetector.load(model_dir)
    log(_markdown(out))
    return det, out


def _markdown(out) -> str:
    L = [f"# Model comparison - selected: **{out['selected']}**", "", f"Selection rule: {out['selection_rule']}", "",
         "| model | status | val worst-group TPR | val TPR fluent | val TPR disfluent | val TPR real voices | gate | val AUC | verifier | test FRR | test FAR | test FA/hour | test AUC |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["candidates"]:
        if r["status"] != "ok":
            L.append(f"| {r['name']} | {r['status']} | | | | | | | | | | | |")
        else:
            L.append(f"| {r['name']}{' (SELECTED)' if r['name'] == out['selected'] else ''} | ok | {r['val_TPR_at_FAR']} | "
                     f"{r.get('val_TPR_fluent', '')} | {r.get('val_TPR_disfluent', '')} | {r.get('val_TPR_real', '')} | {r.get('unit_gate', '')} | "
                     f"{r['val_AUC']} | {r['verifier']} | "
                     f"{r.get('test_FRR', '')} | {r.get('test_FAR', '')} | {r.get('test_false_accepts_per_hour', '')} | {r.get('test_AUC', '')} |")
    return "\n".join(L) + "\n"

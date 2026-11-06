#!/usr/bin/env python3
"""Command line entry point.   python run.py --help"""
from __future__ import annotations

import argparse
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):          # Windows consoles default to cp1252: never crash on a non-ASCII character
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def cmd_check(a):
    print(f"Python {sys.version.split()[0]}  ({sys.executable})")
    ok = sys.version_info >= (3, 9)
    print("  python >= 3.9 ........", "OK" if ok else "TOO OLD")
    for mod, need in (("numpy", True), ("scipy", True), ("sklearn", True), ("flask", True), ("onnxruntime", True), ("matplotlib", False), ("torch", False)):
        try:
            m = importlib.import_module(mod)
            try:
                from importlib.metadata import version
                ver = version({"sklearn": "scikit-learn"}.get(mod, mod))
            except Exception:
                ver = ""
            print(f"  {mod:11s} .......... OK {ver}")
        except Exception:
            print(f"  {mod:11s} .......... {'MISSING (required)' if need else 'not installed (optional; needed only for training)'}")
            ok = ok and not need
    from wakeword.audio_io import ffmpeg_available
    print("  ffmpeg ............", "OK" if ffmpeg_available() else "not found (needed only to decode mp3/ogg/aac/opus uploads)")
    raw = Path(a.raw)
    zips = sorted(p.name for p in raw.glob("*.zip")) if raw.exists() else []
    print(f"  raw data folder {raw}: {len(zips)} zip file(s)")
    for z in zips:
        print("     -", z)
    mdir = Path(a.models)
    if (mdir / "frame_model.npz").exists() or (mdir / "frame_model.pt").exists():
        try:                                   # really load it: "file exists" is not the same as "usable"
            from wakeword.detector import WakeWordDetector
            det = WakeWordDetector.load(mdir)
            print(f"  trained model ...... OK, loads ({(det.meta or {}).get('name', '?')}, threshold {det.threshold:.3f})")
            backend = "ONNX Runtime" if type(det.model).__name__.startswith("Onnx") else "PyTorch"
            print(f"  model backend ...... {backend}" + ("" if backend == "ONNX Runtime" else "  (run scripts/export_onnx.py to run without PyTorch)"))
        except Exception as e:
            print(f"  trained model ...... FOUND BUT NOT LOADABLE: {type(e).__name__}: {e}")
            ok = False
    else:
        print("  trained model ...... not yet (run: python run.py all)")
    mf = Path(a.data) / "manifest.csv"
    if not mf.exists():
        print("  dataset ............ not yet (run: python run.py build-data  or  all)")
    else:
        n = sum(1 for _ in open(mf, encoding="utf-8")) - 1
        if n < 2500:
            print(f"  dataset ............ DEMO SUBSET only ({n} clips) -> run  python run.py build-data  before training")
        else:
            print(f"  dataset ............ full ({n} clips)")
    return 0 if ok else 1


def cmd_audit(a):
    from wakeword.dataset_builder import audit_dataset
    return 1 if audit_dataset(a.data) else 0


def cmd_build(a):
    from wakeword import dataset_builder
    refiner = None
    if a.align and (Path(a.models) / "meta.json").exists():
        from wakeword.detector import WakeWordDetector
        refiner = WakeWordDetector.load(a.models).model
        print("using the current model for forced alignment of the fluent sources")
    dataset_builder.build_dataset(a.data, a.raw, scale=0.35 if a.fast else 1.0, refiner=refiner,
                                  sep28k_root=a.sep28k or None, n_sep=a.n_sep, pos_scale=a.pos_scale, pos_share=a.pos_share,
                                  n_piper=a.n_piper)


def cmd_all(a):
    from wakeword import pipeline
    names = a.models_list.split(",") if a.models_list else None
    pipeline.run_all(a.raw, a.data, a.models, names=names, fast=a.fast, epochs_mlp=a.epochs, epochs_torch=a.epochs_torch,
                     sep28k_root=a.sep28k or None, torch_max_clips=a.torch_clips or None, n_sep=a.n_sep,
                     pos_scale=a.pos_scale, pos_share=a.pos_share, target_far=a.select_far, n_piper=a.n_piper,
                     bootstrap_model=a.bootstrap_model or None, init_from=a.init_from or None, lr=a.lr)


def cmd_compare(a):
    from wakeword.compare import run_comparison
    names = a.models_list.split(",") if a.models_list else None
    run_comparison(a.data, a.models, names=names, epochs_mlp=a.epochs, epochs_torch=a.epochs_torch,
                   torch_max_clips=a.torch_clips or None, force=a.force, target_far=a.select_far,
                   init_from=a.init_from or None, lr=a.lr)


def cmd_eval(a):
    from wakeword.detector import WakeWordDetector
    from wakeword.evaluate import evaluate
    evaluate(WakeWordDetector.load(a.models), a.data, a.split, a.models)


def cmd_detect(a):
    import json
    from wakeword import audio_io
    from wakeword.detector import WakeWordDetector
    det = WakeWordDetector.load(a.models)
    res = det.analyze(audio_io.load_audio(a.file), repair=False, with_curves=False)
    print(json.dumps({k: v for k, v in res.items() if k != "curves"}, indent=2))


def cmd_refit_verifier(a):
    from wakeword.compare import refit_verifier
    refit_verifier(a.models, a.data, target_far=a.select_far)


def cmd_set_threshold(a):
    """Move the decision threshold of the installed model (more recall <-> fewer false wake-ups) without retraining."""
    import json as _json
    vp = Path(a.models) / "verifier.json"
    d = _json.loads(vp.read_text(encoding="utf-8"))
    print(f"threshold {d['threshold']:.3f} -> {a.value:.3f}")
    d["threshold"] = float(a.value)
    vp.write_text(_json.dumps(d), encoding="utf-8")


def cmd_real_prepare(a):
    from wakeword import realpeople
    realpeople.prepare(a.src, fill_silence=a.fill_silence)


def cmd_real_eval(a):
    from wakeword import realpeople
    from wakeword.detector import WakeWordDetector
    realpeople.evaluate(WakeWordDetector.load(a.models), tag=a.tag)


def cmd_real_cv(a):
    from wakeword import realpeople, realtrain
    realtrain.cross_validate(a.models, a.replay, realpeople.REAL_DIR, k=a.folds, epochs=a.epochs, per_epoch=a.per_epoch,
                             out=a.models)


def cmd_real_final(a):
    from wakeword import realpeople, realtrain
    realtrain.final_train(a.models, a.replay, realpeople.REAL_DIR, arm=a.arm, epochs=a.epochs, per_epoch=a.per_epoch, target_far=a.far)


def _neg_streams(a):
    import numpy as np
    from wakeword import farh
    files = farh.list_audio(a.negatives)
    if not files:
        raise SystemExit(f"no audio files under {a.negatives}  (run: python scripts/fetch_negatives.py --out {a.negatives})")
    print(f"  {len(files)} negative audio files under {a.negatives}")
    gen = farh.make_pool_streams(files, np.random.default_rng(a.seed), minutes=12.0)
    if not getattr(a, "max_hours", 0):
        return gen

    def capped():                                       # stop after --max-hours of audio (time budget)
        used = 0.0
        for x in gen:
            yield x
            used += len(x) / 16000 / 3600
            if used >= a.max_hours:
                print(f"  reached --max-hours {a.max_hours}: stopping at {used:.2f} h")
                return
    return capped()


def cmd_farh_eval(a):
    """FRR at fixed false wake-ups per hour on a large pool of negative audio (the product-style metric)."""
    import numpy as np
    from wakeword import farh, realpeople as rp
    from wakeword.detector import WakeWordDetector
    det = WakeWordDetector.load(a.models)
    hours, sc = farh.candidates(det, _neg_streams(a), thr_floor=a.floor, name="negatives")
    pos = {}
    try:
        R = rp.REAL_DIR
        pos["real stutter (in-sample if the model trained on them)"] = [float(det.analyze(rp.load_clip(r, R), repair=False, with_curves=False)["score"])
                                                                       for r in rp.load_manifest(R) if r["is_wake_word"]]
    except (Exception, SystemExit) as exc:
        print("  (no real recordings found:", exc, ")")
    t = farh.det_table(pos, {"negatives": (hours, sc)})
    md = farh.det_markdown(t, "FRR at fixed false wake-ups per hour")
    print(md)
    if a.set_threshold_for:                             # choose the threshold from the measured false-wake-up rate
        row = next((r for r in t["rows"] if abs(r["target_fa_per_hour"] - a.set_threshold_for) < 1e-9), None)
        if row is None:
            t2 = farh.det_table(pos, {"negatives": (hours, sc)}, targets=(a.set_threshold_for,))
            row = t2["rows"][0]
        need_h = 3.0 / a.set_threshold_for                # rule of three: fewer hours cannot show a rate this low
        if hours < need_h:
            print(f"  pool too small ({hours:.2f} h < {need_h:.1f} h needed to show <= {a.set_threshold_for}/h): threshold left unchanged")
        elif row["threshold"] is None:
            print(f"  target {a.set_threshold_for}/h not reachable on this pool: threshold left unchanged")
        elif row["threshold"] < 0.5:
            print(f"  threshold {row['threshold']:.2f} would be below the 0.50 safety floor: left unchanged")
        else:
            cmd_set_threshold(argparse.Namespace(value=row["threshold"], models=a.models))
            print(f"  threshold set so that the measured false-wake-up rate is <= {a.set_threshold_for}/h on this pool "
                  f"(pool = {hours:.2f} h, so the rate is only resolved to {1 / hours:.2f}/h; chosen on the same audio it is optimistic)")
    out = Path(a.models)
    farh.save(t, out / "farh_report.json"); (out / "farh_report.md").write_text(md, encoding="utf-8")
    print("saved", out / "farh_report.md")


def cmd_mine(a):
    """Hard-negative mining: replay negative audio, collect the false wake-ups, refit the decision layer with them."""
    from wakeword import farh, realpeople, realtrain
    from wakeword.detector import WakeWordDetector
    det = WakeWordDetector.load(a.models)
    mined = farh.mine(det, _neg_streams(a), thr_floor=a.floor, name="negatives")
    print(f"  mined {len(mined)} false triggers  (the negative audio must contain NO 'Hey Google')")
    if not a.apply or not mined:
        return
    realtrain.final_train(a.models, a.data, realpeople.REAL_DIR, arm="A", target_far=a.far, hard_negs=[w for w, _ in mined])
    print("  decision layer refitted with the mined negatives; now re-run farh-eval on audio that was NOT used for mining and set the threshold with  python run.py set-threshold VALUE")


def cmd_diagnose(a):
    from wakeword import diagnose
    from wakeword.detector import WakeWordDetector
    if not a.team and not a.path:
        raise SystemExit("give a file / folder, or --team")
    diagnose.run(WakeWordDetector.load(a.models), a.path, team=a.team, raw_dir=a.raw)


def cmd_sep28k(a):
    from wakeword import sep28k
    if a.cmd == "sep28k-scan":
        sep28k.scan(a.path)
    elif a.cmd == "sep28k-calibrate":
        sep28k.calibrate(a.path, max_clips=a.max_clips)
    else:
        from wakeword.detector import WakeWordDetector
        det = WakeWordDetector.load(a.models)
        if a.tune is not None:
            sep28k.tune_threshold(det, a.path, a.models, a.tune, max_clips=a.max_clips)
        else:
            sep28k.evaluate(det, a.path, max_clips=a.max_clips, out_dir=a.models, part=a.part)


def cmd_serve(a):
    from server.app import create_app
    app = create_app(a.models, a.data)
    print(f"\n  Open  http://{a.host}:{a.port}   in Chrome / Edge / Firefox\n")
    app.run(host=a.host, port=a.port, debug=False, threaded=True)


def cmd_selftest_torch(a):
    from wakeword import torch_models
    sys.exit(0 if torch_models.selftest() else 1)


def cmd_test(a):
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)


def main():
    p = argparse.ArgumentParser(description="Wake-word recognition for speech disfluency")
    p.add_argument("--raw", default=str(ROOT / "raw"), help="folder with the .zip datasets")
    p.add_argument("--data", default=str(ROOT / "data"), help="where the generated dataset lives")
    p.add_argument("--models", default=str(ROOT / "models"), help="where the trained model lives")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="check the environment").set_defaults(f=cmd_check)
    s = sub.add_parser("build-data", help="generate the fluent / stuttered / negative dataset")
    s.add_argument("--fast", action="store_true", help="small quick dataset (about 1/3)")
    s.add_argument("--align", action="store_true", help="use the current model for forced alignment")
    s.add_argument("--sep28k", default="", help="folder or .zip with the full SEP-28k clips (real stuttering -> extra negatives)")
    s.add_argument("--n-sep", type=int, default=0, help="how many SEP-28k clips to use as negatives (0 = as many as the positive share needs)")
    s.add_argument("--pos-scale", type=float, default=1.0, help="multiplies the number of fluent / stuttered clips made per recording")
    s.add_argument("--pos-share", type=float, default=0.20, help="share of wake-word clips in EVERY split (train, val and test)")
    s.add_argument("--n-piper", type=int, default=1000, help="generated voices (10 emotions x child/teen/adult/elderly) used as wake-word sources")
    s.set_defaults(f=cmd_build)
    for name, fn, hlp in (("all", cmd_all, "build data, train + compare models, keep the best (the one-command route)"),
                          ("compare", cmd_compare, "train + compare models on the existing dataset and keep the best")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--models-list", default="", help="comma list, e.g. mlp_small,mlp_large,cnn,crnn (default: all available)")
        s.add_argument("--epochs", type=int, default=12, help="epochs for the scikit-learn MLPs")
        s.add_argument("--epochs-torch", type=int, default=30, help="epochs for the PyTorch models")
        s.add_argument("--fast", action="store_true", help="quick smoke run (small data, few epochs)")
        s.add_argument("--torch-clips", type=int, default=0, help="train the PyTorch models on only N clips (CPU speed-up; 0 = all)")
        s.add_argument("--sep28k", default="", help="folder or .zip with SEP-28k clips (used by 'all')")
        s.add_argument("--force", action="store_true", help="compare even if the dataset is only the small demo subset")
        s.add_argument("--select-far", type=float, default=0.01,
                       help="false accept rate on the validation negatives used to pick the winner (default 0.01; the report also lists 1/2/5/10%%)")
        s.add_argument("--init-from", default="", help="fine-tune the PyTorch models from a trained model folder (e.g. models_v12)")
        s.add_argument("--lr", type=float, default=None, help="learning rate for the PyTorch models (default 2e-3, 5e-4 when fine-tuning)")
        s.set_defaults(f=fn)
    s = sub.add_parser("evaluate", help="evaluate the selected model on a split")
    s.add_argument("--split", default="test")
    s.set_defaults(f=cmd_eval)
    s = sub.add_parser("detect", help="run the detector on one audio file")
    s.add_argument("file")
    s.set_defaults(f=cmd_detect)
    s = sub.add_parser("refit-verifier", help="re-choose and refit only the verifier on the installed model (minutes, no retraining)")
    s.add_argument("--select-far", type=float, default=0.01)
    s.set_defaults(f=cmd_refit_verifier)
    s = sub.add_parser("diagnose", help="explain why clips are (not) detected: a file, a folder, or --team")
    s.add_argument("path", nargs="?", default="")
    s.add_argument("--team", action="store_true", help="run over the stuttered clips of the project datasets")
    s.set_defaults(f=cmd_diagnose)
    s = sub.add_parser("serve", help="start backend + frontend")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(f=cmd_serve)
    for name, hlp in (("sep28k-scan", "inspect your SEP-28k folder / zip"),
                      ("sep28k-eval", "false accepts per hour of REAL stuttered speech"),
                      ("sep28k-calibrate", "measure real gap / prolongation / block durations -> calibration.json")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("path", help="folder or .zip that holds the SEP-28k clips (and the label CSV)")
        s.add_argument("--max-clips", type=int, default=5000)
        s.set_defaults(f=cmd_sep28k)
    ev = sub.choices["sep28k-eval"]
    ev.add_argument("--part", default="test", choices=["test", "dev", "train", "all"],
                    help="which SEP-28k shows to evaluate on (default: the held-out TEST shows)")
    ev.add_argument("--tune", type=float, default=None, metavar="FA_PER_HOUR",
                    help="set the threshold on the DEV shows so that at most this many false accepts per hour remain, then stop")
    for opt, kw in (("--n-sep", dict(type=int, default=0, help="how many SEP-28k clips to use as negatives (0 = as many as the positive share needs)")),
                    ("--pos-scale", dict(type=float, default=1.0, help="multiplies the number of fluent / stuttered clips made per recording")),
                    ("--pos-share", dict(type=float, default=0.20, help="share of wake-word clips in EVERY split (train, val and test)")),
                    ("--n-piper", dict(type=int, default=1000, help="generated voices (emotions x age groups) used as wake-word sources")),
                    ("--bootstrap-model", dict(default="", help="trained model folder used to verify + align the recordings instead of a quick first-pass model"))):
        sub.choices["all"].add_argument(opt, **kw)
    for name, fn, hlp in (("farh-eval", cmd_farh_eval, "FRR at fixed false wake-ups per hour on a folder of negative audio"),
                          ("mine-negatives", cmd_mine, "replay negative audio, collect false wake-ups, optionally refit the decision layer (--apply)")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--negatives", default=str(ROOT / "negatives"), help="folder of audio WITHOUT the wake word (see scripts/fetch_negatives.py)")
        s.add_argument("--floor", type=float, default=0.3, help="lowest score recorded as a candidate wake-up")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--max-hours", type=float, default=0.0, help="use at most this many hours of the audio (0 = all)")
        if name == "farh-eval":
            s.add_argument("--set-threshold-for", type=float, default=0.0, help="set the model threshold for this many false wake-ups per hour (e.g. 1.0)")
        if name == "mine-negatives":
            s.add_argument("--apply", action="store_true"); s.add_argument("--far", type=float, default=0.01)
        s.set_defaults(f=fn)
    s = sub.add_parser("set-threshold", help="move the decision threshold of the installed model (lower = more wake-ups accepted)")
    s.add_argument("value", type=float); s.add_argument("--models", default=str(ROOT / "models"))
    s.set_defaults(f=cmd_set_threshold)
    s = sub.add_parser("real-prepare", help="clean the recordings of real people who stutter (any length / format) -> real_people/")
    s.add_argument("src", help="folder with <phrase>_spk<N>.<ext> recordings, e.g. hey_google_spk3.ogg")
    s.add_argument("--fill-silence", action="store_true", help="replace digital silence by a faint noise floor")
    s.set_defaults(f=cmd_real_prepare)
    s = sub.add_parser("real-eval", help="score the installed model on the cleaned real recordings")
    s.add_argument("--models", default=str(ROOT / "models")); s.add_argument("--tag", default="current")
    s.set_defaults(f=cmd_real_eval)
    for name, hlp, fn in (("real-cv", "leave-speakers-out cross-validation: baseline vs verifier-only vs fine-tuning", cmd_real_cv),
                          ("real-final", "fine-tune on ALL real speakers and install (previous model is backed up)", cmd_real_final)):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--models", default=str(ROOT / "models"))
        s.add_argument("--replay", default=str(ROOT / "data"), help="the original dataset, replayed so the model does not forget it")
        s.add_argument("--epochs", type=int, default=6); s.add_argument("--per-epoch", type=int, default=360)
        if name == "real-cv":
            s.add_argument("--folds", type=int, default=7)
        else:
            s.add_argument("--far", type=float, default=0.03, help="target false-accept rate used to place the threshold")
            s.add_argument("--arm", choices=["A", "B"], default="B", help="A = refit verifier only, B = fine-tune + verifier")
        s.set_defaults(f=fn)
    s = sub.add_parser("audit-data", help="check the dataset: labels, splits, shares, files, wording")
    s.set_defaults(f=cmd_audit)
    sub.add_parser("selftest-torch", help="check that the PyTorch models run on this machine").set_defaults(f=cmd_selftest_torch)
    sub.add_parser("test", help="run the unit tests").set_defaults(f=cmd_test)
    a = p.parse_args()
    a.cmd = a.cmd
    if not hasattr(a, "fast"):
        a.fast = False
    sys.exit(a.f(a) or 0)


if __name__ == "__main__":
    main()

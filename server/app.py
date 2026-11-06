"""Flask backend: wake-word detection API, dataset explorer API, disfluency lab API + static frontend."""
from __future__ import annotations

import base64
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wakeword import audio_io, segmenter  # noqa: E402
from wakeword.config import CLASSES, DATA_DIR, FRONTEND_DIR, HOP, MODEL_DIR, SR  # noqa: E402
from wakeword.detector import WakeWordDetector  # noqa: E402
from wakeword.disfluency import DISFLUENCY_TYPES, DisfluencySynthesizer, random_spec  # noqa: E402
from wakeword.features import logmel  # noqa: E402
from wakeword.model import read_annotations, read_manifest  # noqa: E402
from wakeword.sources import _unit_sane  # noqa: E402

MAX_ANALYSIS_S = 15.0


def _b64(buf: bytes) -> str:
    return base64.b64encode(buf).decode("ascii")


def create_app(model_dir=MODEL_DIR, data_dir=DATA_DIR) -> Flask:
    model_dir, data_dir = Path(model_dir), Path(data_dir)
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
    # Public mode (default): only the Listen and Results views. The research views stream dataset clips and list results on
    # the volunteers' recordings, so they are switched on only for local use with the environment variable WW_RESEARCH=1.
    research = os.environ.get("WW_RESEARCH", "0").strip() == "1"
    _RESEARCH_ONLY = ("/api/dataset/", "/api/synthesize", "/api/live/", "/api/model/comparison")
    lock = threading.Lock()
    cache: dict = {"det": None, "det_mtime": None, "rows": None, "ann": None, "ds_mtime": None, "err": None}

    # ------------------------------------------------------------------ lazy resources
    def get_detector():
        # the selected model is frame_model.npz (MLP) or frame_model.pt (CNN / CRNN)
        p = next((model_dir / n for n in ("frame_model.npz", "frame_model.pt") if (model_dir / n).exists()), None)
        if p is None:
            cache["err"] = None
            return None
        mt = (p.stat().st_mtime, (model_dir / "verifier.json").stat().st_mtime if (model_dir / "verifier.json").exists() else 0)
        with lock:
            if cache["det"] is None or cache["det_mtime"] != mt:
                try:
                    cache["det"], cache["err"] = WakeWordDetector.load(model_dir), None
                except Exception as e:           # a damaged or incompatible model file: say so, do not claim "no model"
                    cache["det"], cache["err"] = None, f"{type(e).__name__}: {e}"
                cache["det_mtime"] = mt
            return cache["det"]

    def get_dataset():
        p = data_dir / "manifest.csv"
        if not p.exists():
            return None, None
        with lock:
            if cache["rows"] is None or cache["ds_mtime"] != p.stat().st_mtime:
                cache["rows"] = read_manifest(data_dir)
                cache["ann"] = read_annotations(data_dir)
                cache["ds_mtime"] = p.stat().st_mtime
            return cache["rows"], cache["ann"]

    def read_request_audio() -> np.ndarray:
        if "audio" in request.files:
            buf = request.files["audio"].read()
        else:
            buf = request.get_data()
        x = audio_io.load_audio_bytes(buf)
        if len(x) > MAX_ANALYSIS_S * SR:
            x = x[-int(MAX_ANALYSIS_S * SR):]
        return x

    @app.errorhandler(audio_io.AudioDecodeError)
    def _decode_err(e):
        return jsonify({"error": str(e)}), 400

    @app.errorhandler(Exception)
    def _unhandled(e):
        """Log the full error on the server; send visitors a plain message without paths or internals."""
        from werkzeug.exceptions import HTTPException
        if isinstance(e, HTTPException):
            return e
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Something went wrong on the server. Please try again."}), 500

    @app.errorhandler(413)
    def _too_big(e):
        return jsonify({"error": "audio too large (max 40 MB)"}), 413

    @app.before_request
    def _public_mode():
        if not research and request.path.startswith(_RESEARCH_ONLY):
            return jsonify({"error": "Not available."}), 404
        return None

    # ------------------------------------------------------------------ frontend
    @app.get("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.get("/static/<path:name>")
    def static_files(name):
        return send_from_directory(FRONTEND_DIR, name)

    # ------------------------------------------------------------------ status
    @app.get("/api/health")
    def health():
        det = get_detector()
        rows, _ = get_dataset()
        return jsonify({"status": "ok", "model_loaded": det is not None, "model_name": (det.meta or {}).get("name") if det else None,
                        "threshold": det.threshold if det else None,
                        "model_error": bool(cache["err"]), "dataset_loaded": rows is not None, "n_samples": len(rows) if rows else 0,
                        "ffmpeg": audio_io.ffmpeg_available(), "keyword": "Hey Google", "research": research})

    # ------------------------------------------------------------------ detection
    @app.post("/api/detect")
    def detect():
        det = get_detector()
        if det is None:
            if cache["err"]:
                print(f"[server] model in {model_dir} could not be loaded: {cache['err']}", flush=True)
                return jsonify({"error": "The detection model could not be loaded. Please try again later."}), 503
            return jsonify({"error": "The detection model is not available on this server."}), 503
        x = read_request_audio()
        want_repair = request.args.get("repair", "0") == "1"
        res = det.analyze(x, repair=want_repair)
        rep = res.pop("repaired_audio", None)
        if rep is not None:
            res["repaired_wav_b64"] = _b64(audio_io.wav_bytes(rep))
        return jsonify(res)

    # ------------------------------------------------------------------ dataset explorer
    @app.get("/api/dataset/summary")
    def ds_summary():
        rows, _ = get_dataset()
        if rows is None:
            return jsonify({"error": "The dataset is not available on this server."}), 404
        from collections import Counter
        out = {"total": len(rows), "by_split": Counter(r["split"] for r in rows),
               "by_label": Counter(r["label"] for r in rows),
               "by_category": Counter(r["category"] for r in rows),
               "by_speaker_kind": Counter(r["speaker_kind"] for r in rows),
               "by_stutter_profile": Counter(r.get("stutter_profile", "n/a") for r in rows if r["label"] == "disfluent"),
               "by_emotion": Counter(r.get("emotion", "n/a") for r in rows if r["is_wake_word"] == "1"),
               "by_age_group": Counter(r.get("age_group", "n/a") for r in rows if r["is_wake_word"] == "1"),
               "hours": round(sum(float(r["duration_seconds"]) for r in rows) / 3600, 3)}
        return jsonify(out)

    @app.get("/api/dataset/samples")
    def ds_samples():
        rows, _ = get_dataset()
        if rows is None:
            return jsonify({"error": "The dataset is not available on this server."}), 404
        split, label, cat = request.args.get("split"), request.args.get("label"), request.args.get("category")
        q = (request.args.get("q") or "").lower()
        page, per = max(int(request.args.get("page", 1)), 1), min(int(request.args.get("per_page", 40)), 200)
        sel = [r for r in rows if (not split or r["split"] == split) and (not label or r["label"] == label)
               and (not cat or r["category"] == cat or r["category"].startswith(cat + ":"))
               and (not q or q in r["sample_id"].lower() or q in r["category"].lower() or q in r["source_id"].lower())]
        cats = sorted({r["category"] for r in rows})
        slim = [{k: r.get(k, "") for k in ("sample_id", "split", "label", "category", "affected_word", "duration_seconds",
                                           "speaker_kind", "n_events", "source_id", "context", "stutter_profile",
                                           "emotion", "age_group")} for r in sel[(page - 1) * per: page * per]]
        return jsonify({"total": len(sel), "page": page, "per_page": per, "items": slim, "categories": cats})

    def _find(sample_id):
        rows, ann = get_dataset()
        if rows is None:
            abort(404)
        for r in rows:
            if r["sample_id"] == sample_id:
                return r, ann.get(sample_id, {})
        abort(404)

    @app.get("/api/dataset/audio/<sample_id>")
    def ds_audio(sample_id):
        r, _ = _find(sample_id)
        path = (data_dir / r["relative_path"]).resolve()
        if data_dir.resolve() not in path.parents or not path.exists():
            abort(404)
        return send_file(path, mimetype="audio/wav")

    @app.get("/api/dataset/sample/<sample_id>")
    def ds_sample(sample_id):
        rows, _ = get_dataset()
        r, a = _find(sample_id)
        x = audio_io.load_audio(data_dir / r["relative_path"])
        n = 700
        step = max(1, len(x) // n)
        trimmed = x[: step * n].reshape(-1, step) if len(x) >= step * n else x[: (len(x) // step) * step].reshape(-1, step)
        peaks = [[round(float(c.min()), 3), round(float(c.max()), 3)] for c in trimmed]
        lm = logmel(x)
        lo, hi = np.percentile(lm, 2), np.percentile(lm, 99.5)
        img = np.clip((lm - lo) / max(hi - lo, 1e-6), 0, 1)
        pair = None
        if r["label"] == "disfluent":
            for q in rows:
                if q["label"] == "fluent" and q["source_id"] == r["source_id"]:
                    pair = q["sample_id"]
                    break
        return jsonify({"meta": r, "segments": a.get("segments", []), "events": a.get("events", []),
                        "classes": CLASSES, "duration": len(x) / SR, "peaks": peaks,
                        "spec": {"frames": int(img.shape[0]), "bins": int(img.shape[1]),
                                 "b64": _b64((img * 255).astype(np.uint8).tobytes()), "hop_s": HOP / SR},
                        "fluent_pair": pair})

    # ------------------------------------------------------------------ disfluency lab ("mimic this voice")
    @app.post("/api/synthesize")
    def synthesize():
        x = read_request_audio()
        win = segmenter.pick_wake_word_window(x, None, max_len=3.0)
        if win is None:
            return jsonify({"error": "No speech found. Record a clear, fluent 'Hey Google'."}), 422
        u = segmenter.build_utterance(x, (win[0], win[1]))
        if u is None or not _unit_sane(u):
            return jsonify({"error": "Could not split the recording into Hey / Goo / gle. "
                                     "Say 'Hey Google' once, clearly, with a short pause before and after."}), 422
        types = [t for t in (request.values.get("types") or ",".join(DISFLUENCY_TYPES)).split(",") if t in DISFLUENCY_TYPES]
        types = types or DISFLUENCY_TYPES
        per = int(np.clip(int(request.values.get("n", 1)), 1, 3))
        seed = int(request.values.get("seed", 7))
        rng = np.random.default_rng(seed)
        out = []
        base = DisfluencySynthesizer(u, rng).build({"type": "fluent"})
        out.append({"type": "fluent", "events": [], "duration": len(base.audio) / SR, "segments": base.segments,
                    "wav_b64": _b64(audio_io.wav_bytes(base.audio))})
        for t in types:
            for _ in range(per):
                s = DisfluencySynthesizer(u, np.random.default_rng(int(rng.integers(0, 2**31 - 1)))).build(random_spec(rng, t))
                out.append({"type": t, "events": s.events, "duration": len(s.audio) / SR,
                            "segments": [[round(a, 3), round(b, 3), int(c), tg] for a, b, c, tg in s.segments],
                            "wav_b64": _b64(audio_io.wav_bytes(s.audio))})
        return jsonify({"items": out, "classes": CLASSES})

    # ------------------------------------------------------------------ report
    @app.get("/api/model/report")
    def report():
        p = model_dir / "metrics.json"
        if not p.exists():
            return jsonify({"error": "Evaluation results are not available on this server."}), 404
        d = json.loads(p.read_text(encoding="utf-8"))
        mp = next((model_dir / n for n in ("frame_model.pt", "frame_model.npz") if (model_dir / n).exists()), None)
        d["report_time"] = _iso(p)
        d["model_time"] = _iso(mp) if mp else None
        d["report_older_than_model"] = bool(mp and p.stat().st_mtime < mp.stat().st_mtime - 1)
        return jsonify(d)

    @app.get("/api/model/comparison")
    def comparison():
        p = model_dir / "comparison.json"
        if not p.exists():
            return jsonify({"error": "Model comparison is not available on this server."}), 404
        return jsonify(json.loads(p.read_text(encoding="utf-8")))


    # ------------------------------------------------------------------ live evaluation on the real recordings
    def _iso(p):
        import datetime as _dt
        return _dt.datetime.fromtimestamp(Path(p).stat().st_mtime).strftime("%Y-%m-%d %H:%M") if Path(p).exists() else None

    @app.get("/api/live/eval")
    def live_eval():
        """Re-score EVERY real recording with the model that is loaded right now (nothing is read from an old report)."""
        from wakeword import realpeople as rp
        det = get_detector()
        if det is None:
            return jsonify({"error": "no model loaded"}), 503
        try:
            rows = rp.load_manifest(rp.REAL_DIR)
        except (Exception, SystemExit) as e:
            return jsonify({"error": "Real recordings are not available on this server."}), 404
        key = (cache["det_mtime"], len(rows))
        with lock:
            st = cache.get("le_state")
            if not st or st["key"] != key:
                st = {"key": key, "done": 0, "total": len(rows), "res": None, "err": None}
                cache["le_state"] = st

                def work(st=st, det=det, rows=rows):
                    try:
                        clips = []
                        for r in rows:
                            x = rp.load_clip(r, rp.REAL_DIR)
                            a = det.analyze(x, repair=False, with_curves=False)
                            clips.append({"id": r["sample_id"], "speaker": r["speaker"], "phrase": r["phrase"], "wake": int(r["is_wake_word"]),
                                          "score": a["score"], "detected": a["detected"], "reason": a.get("reason", ""), "heard": a.get("heard_text", ""),
                                          "events": [e["type"] for e in a.get("events", [])], "dur": round(len(x) / SR, 2)})
                            st["done"] = len(clips)
                        st["res"] = {"clips": clips}
                    except Exception as exc:
                        st["err"] = f"{type(exc).__name__}: {exc}"
                threading.Thread(target=work, daemon=True).start()
        if st["err"]:
            print(f"[server] live evaluation failed: {st['err']}", flush=True)
            return jsonify({"error": "The live check could not be computed on this server."}), 500
        if st["res"] is None:
            return jsonify({"status": "running", "done": st["done"], "total": st["total"]}), 202
        res = st["res"]
        pos = [c["score"] for c in res["clips"] if c["wake"]]
        neg = [c["score"] for c in res["clips"] if not c["wake"]]
        sweep = [{"t": round(t, 3), "recall": float(np.mean([v >= t for v in pos])) if pos else None,
                  "far": float(np.mean([v >= t for v in neg])) if neg else None} for t in np.arange(0, 1.0001, 0.02)]
        auc = rp._auc([1] * len(pos) + [0] * len(neg), pos + neg) if pos and neg else None
        fr = model_dir / "farh_report.json"
        farh = json.loads(fr.read_text(encoding="utf-8")) if fr.exists() else None
        mp = next((model_dir / n for n in ("frame_model.pt", "frame_model.npz") if (model_dir / n).exists()), None)
        return jsonify({"model": {"type": (det.meta or {}).get("frame_type") or (det.meta or {}).get("name"), "threshold": det.threshold,
                                  "soft_margin": __import__("wakeword.detector", fromlist=["SOFT_MARGIN"]).SOFT_MARGIN,
                                  "model_file_time": _iso(mp) if mp else None, "verifier_time": _iso(model_dir / "verifier.json")},
                        "clips": res["clips"], "sweep": sweep, "auc": auc, "n_pos": len(pos), "n_neg": len(neg),
                        "farh": farh, "farh_time": _iso(fr),
                        "farh_stale": bool(farh and mp and fr.stat().st_mtime < mp.stat().st_mtime - 1)})

    @app.get("/api/live/heldout")
    def live_heldout():
        """Leave-speakers-out cross-validation: every speaker scored by a model that never heard them (from models/cv_report.json)."""
        p = model_dir / "cv_report.json"
        if not p.exists():
            return jsonify({"error": "Cross-validation results are not available on this server."}), 404
        j = json.loads(p.read_text(encoding="utf-8"))
        arm = j["summary"].get("B") or next(v for k, v in j["summary"].items() if k != "baseline")
        return jsonify({"time": _iso(p), "k": j.get("k"), "arm": arm.get("tag"), "auc": arm.get("AUC"), "clips": arm.get("clips", []),
                        "sweep": arm.get("threshold_sweep", []), "baseline_auc": j["summary"].get("baseline", {}).get("AUC")})

    @app.get("/api/live/patterns")
    def live_patterns():
        """Stuttering pattern measured on the real recordings (acoustic detectors in wakeword/patterns.py)."""
        from wakeword import patterns as P, realpeople as rp
        try:
            rows = rp.load_manifest(rp.REAL_DIR)
        except (Exception, SystemExit) as e:
            return jsonify({"error": "Real recordings are not available on this server."}), 404
        key = ("pat", len(rows))
        with lock:
            hit = cache.get("live_pat")
        if hit and hit[0] == key:
            return jsonify(hit[1])
        out = []
        for r in rows:
            x = rp.load_clip(r, rp.REAL_DIR)
            e = P.acoustic_events(x); st = P.clip_stats(x); rs = P.restarts(x)
            out.append({"id": r["sample_id"], "speaker": r["speaker"], "phrase": r["phrase"],
                        "span": None if st is None else round(float(st[0]), 2), "n_voiced": e["n_seg"],
                        "blocks": e["blocks"], "pauses": e["pauses"], "reps": e["reps"], "prolong": e["prolong"],
                        "restarts": len(rs), "segs": [[round(a, 2), round(b, 2)] for a, b in e["segs"]]})
        res = {"clips": out}
        with lock:
            cache["live_pat"] = (key, res)
        return jsonify(res)

    def _warm():                                  # load models + run one analysis now, so the FIRST utterance is not missed
        try:
            det = get_detector()
            if det is not None:
                det.analyze((np.random.default_rng(0).standard_normal(2 * SR) * 0.05).astype(np.float32), repair=False, with_curves=False)
        except Exception:
            pass
    threading.Thread(target=_warm, daemon=True).start()
    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=8000, debug=False)

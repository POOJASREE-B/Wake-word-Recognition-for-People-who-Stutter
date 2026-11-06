"""Regression tests for the problems found on a real Windows machine + the SEP-28k tools."""
import csv
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                   # noqa: E402

from tests.test_core import fake_hey_google          # noqa: E402
from wakeword import audio_io, sep28k                # noqa: E402


class TestEncoding(unittest.TestCase):
    """Windows opens text files as cp1252 unless told otherwise -> every text read/write must say utf-8."""

    def test_all_text_io_declares_encoding(self):
        bad = []
        for py in list((ROOT / "wakeword").glob("*.py")) + list((ROOT / "server").glob("*.py")) + [ROOT / "run.py"]:
            for n, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r"\.(read_text|write_text)\(", line) and "encoding" not in line:
                    bad.append(f"{py.name}:{n}")
                if re.search(r"\bopen\(", line) and "encoding" not in line and not re.search(r"""["'][rwa]b["']|zf\.open|ZipFile|urlopen""", line):
                    bad.append(f"{py.name}:{n}")
        self.assertEqual(bad, [], "text I/O without encoding= : " + ", ".join(bad))

    def test_markdown_reports_are_ascii_safe(self):
        from wakeword import compare, evaluate
        out = {"selected": "a", "selection_rule": "r", "candidates": [{"name": "a", "status": "ok", "val_TPR_at_FAR": .5, "val_AUC": .9, "verifier": "lr"}]}
        compare._markdown(out).encode("cp1252")


class TestDemoGuard(unittest.TestCase):
    def test_compare_refuses_tiny_dataset(self):
        from wakeword.compare import run_comparison
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with open(d / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["sample_id", "split", "relative_path", "is_wake_word"])
                w.writeheader()
                for i in range(20):
                    w.writerow({"sample_id": f"S{i}", "split": "train", "relative_path": "x.wav", "is_wake_word": 0})
            (d / "annotations.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                run_comparison(d, d / "m", names=["mlp_small"])
            self.assertIn("DEMO", str(cm.exception))
            self.assertFalse((d / "m" / "frame_model.npz").exists())


class DummyDetector:
    threshold = 0.5

    def analyze(self, x, repair=False, with_curves=False):
        return {"detected": False, "score": 0.1}


class TestSep28k(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        rows = []
        for ep in range(3):
            for clip in range(12):
                x = fake_hey_google(clip)
                audio_io.save_wav(d / "clips" / "ShowA" / str(ep) / f"ShowA_{ep}_{clip}.wav", x[: 3 * 16000])
                ev = ["Block", "SoundRep", "Prolongation", "NoStutteredWords"][clip % 4]
                row = {"Show": "ShowA", "EpId": ep, "ClipId": clip, "Prolongation": 0, "Block": 0, "SoundRep": 0, "WordRep": 0,
                       "Interjection": 0, "NoStutteredWords": 0, "PoorAudioQuality": 0, "Music": 0, "NoSpeech": 0}
                row[ev] = 3
                rows.append(row)
        with open(d / "SEP-28k_labels.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        self.d = d

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan(self):
        r = sep28k.scan(self.d, log=lambda *_: None)
        self.assertEqual(r["n_clips"], 36)
        self.assertEqual(r["n_episodes"], 3)
        self.assertEqual(r["clips_with_labels"], 36)
        self.assertEqual(r["dominant_event_counts"]["Block"], 9)

    def test_scan_zip(self):
        import zipfile
        zp = self.d / "all.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            for p in (self.d / "clips").rglob("*.wav"):
                zf.write(p, p.relative_to(self.d).as_posix())
            zf.write(self.d / "SEP-28k_labels.csv", "SEP-28k_labels.csv")
        r = sep28k.scan(zp, log=lambda *_: None)
        self.assertEqual(r["n_clips"], 36)
        self.assertEqual(r["clips_with_labels"], 36)

    def test_pool_and_eval(self):
        pool = sep28k.load_pool(self.d, 10, log=lambda *_: None)
        self.assertEqual(len(pool), 10)
        res = sep28k.evaluate(DummyDetector(), self.d, max_clips=20, out_dir=self.d / "out", log=lambda *_: None)
        self.assertEqual(res["false_accepts"], 0)
        self.assertTrue((self.d / "out" / "sep28k_eval.json").exists())

    def test_split_shows_keeps_shows_whole(self):
        class K:
            def __init__(self, show):
                self.show = show
        clips = [K(f"show{i}") for i in range(10) for _ in range(5 + i)]
        parts = sep28k.split_shows(clips)
        self.assertEqual(set(parts), {f"show{i}" for i in range(10)})          # one part per show
        self.assertEqual(set(parts.values()), {"train", "dev", "test"})
        self.assertEqual(set(sep28k.split_shows(clips[:3]).values()), {"all"})  # too few shows to split

    def test_threshold_for_rate(self):
        sc = np.sort(np.linspace(0.0, 0.99, 200))[::-1]
        t = sep28k.threshold_for_rate(sc, 1.0, 20)                              # 20 false accepts allowed in 1 h
        self.assertLessEqual(int((sc >= t).sum()), 20)
        self.assertIsNone(sep28k.threshold_for_rate(sc, 1.0, 1))                # 1 allowed: audio too short to say

    def test_tune_refuses_too_little_audio(self):
        md = self.d / "model"
        md.mkdir()
        (md / "verifier.json").write_text('{"threshold": 0.9}', encoding="utf-8")
        with self.assertRaises(SystemExit):
            sep28k.tune_threshold(DummyDetector(), self.d, md, 1.0, log=lambda *_: None)
        self.assertEqual(json.loads((md / "verifier.json").read_text(encoding="utf-8"))["threshold"], 0.9)

    def test_calibrate_writes_file(self):
        out = self.d / "cal.json"
        cal = sep28k.calibrate(self.d, out_file=out, log=lambda *_: None)
        self.assertTrue(out.exists())
        self.assertIn("repetition_gap_ms", cal)


class TestProfilesAndBalance(unittest.TestCase):
    """Stuttering profiles, emotion / age modulation, and the equal wake-word share in every split."""

    def setUp(self):
        from wakeword.segmenter import build_utterance
        self.u = build_utterance(fake_hey_google())
        self.assertIsNotNone(self.u)

    def test_every_profile_and_type_is_labelled_and_builds(self):
        from wakeword.disfluency import DISFLUENCY_TYPES, PROFILE_NAMES, DisfluencySynthesizer, random_spec
        rng = np.random.default_rng(3)
        for prof in PROFILE_NAMES:
            for dtype in [None] + list(DISFLUENCY_TYPES):
                spec = random_spec(rng, dtype, prof, "angry", "child")
                syn = DisfluencySynthesizer(self.u, np.random.default_rng(int(rng.integers(1 << 30)))).build(spec)
                self.assertEqual(syn.label, "disfluent")
                self.assertTrue(np.all(np.isfinite(syn.audio)))
                self.assertEqual(spec["profile"], prof)
                primary = [e for e in syn.events if not e["type"].startswith("secondary")]
                self.assertTrue(primary, (prof, dtype))                          # never a clip with only a secondary event

    def test_emotion_and_age_shift_the_style(self):
        from wakeword.disfluency import style
        self.assertGreater(style("developmental", "angry", "adult")["tension"], style("developmental", "calm", "adult")["tension"])
        self.assertGreater(style("developmental", "sad", "adult")["prolong"], style("developmental", "happy", "adult")["prolong"])
        self.assertEqual(style("developmental", None, "child")["rep_bonus"], 1)

    def test_weighted_split_follows_clip_shares(self):
        from wakeword.dataset_builder import assign_splits_weighted
        w = {f"g{i}": n for i, n in enumerate([900, 700, 500, 400, 300, 200, 100, 100, 50, 50] + [20] * 30)}
        sp = assign_splits_weighted(w, np.random.default_rng(0))
        tot = sum(w.values())
        for name, share in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
            got = sum(n for g, n in w.items() if sp[g] == name) / tot
            self.assertAlmostEqual(got, share, delta=0.03, msg=name)

    def test_neutral_names(self):
        from wakeword.dataset_builder import neutral
        self.assertEqual(neutral("teen_sim"), "teen")
        self.assertEqual(neutral("synthetic_noise"), "generated_noise")
        self.assertEqual(neutral("real_whatsapp"), "real_recording")


if __name__ == "__main__":
    unittest.main()

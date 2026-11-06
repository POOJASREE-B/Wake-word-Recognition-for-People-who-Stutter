"""Unit tests that need no dataset and no trained model."""
import io
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wakeword import audio_io, decoder, dsp, features                      # noqa: E402
from wakeword.config import BG, GLE, GOO, HEY, SR                           # noqa: E402
from wakeword.disfluency import DISFLUENCY_TYPES, DisfluencySynthesizer, random_spec   # noqa: E402
from wakeword.negatives import HARD_KINDS, NOISE_KINDS, hard_negative, noise_negative   # noqa: E402
from wakeword.segmenter import Utterance, build_utterance                   # noqa: E402


def fake_hey_google(seed=0):
    """Synthetic 'Hey Google'-like signal: three voiced bursts separated by short energy dips."""
    rng = np.random.default_rng(seed)
    t = lambda d: np.arange(int(d * SR)) / SR
    def burst(d, f0, a=0.3):
        x = sum(np.sin(2 * np.pi * f0 * k * t(d)) / k for k in range(1, 6)) * np.hanning(int(d * SR) + 2)[1:-1][:len(t(d))]
        return (a * x).astype(np.float32)
    noise = lambda d: (rng.standard_normal(int(d * SR)) * 0.002).astype(np.float32)
    x = np.concatenate([noise(0.4), burst(0.28, 140), noise(0.05), burst(0.30, 150), noise(0.04), burst(0.26, 135), noise(0.5)])
    return x


class TestAudio(unittest.TestCase):
    def test_wav_roundtrip(self):
        x = fake_hey_google()
        y = audio_io.load_audio_bytes(audio_io.wav_bytes(x))
        self.assertEqual(len(x), len(y))
        self.assertLess(float(np.abs(x - y).max()), 1e-3)

    def test_bad_audio_raises(self):
        with self.assertRaises(audio_io.AudioDecodeError):
            audio_io.load_audio_bytes(b"")


class TestDsp(unittest.TestCase):
    def test_wsola_length(self):
        x = fake_hey_google()[8000:12000]
        for st in (0.8, 1.5, 4.0):
            y = dsp.wsola(x, st)
            self.assertAlmostEqual(len(y) / len(x), st, delta=0.05)
            self.assertTrue(np.isfinite(y).all())

    def test_pitch_shift_keeps_length(self):
        x = fake_hey_google()
        self.assertEqual(len(dsp.pitch_shift(x, 2.0)), len(x))

    def test_noise_floor_never_nan(self):
        ref = fake_hey_google()[:3000]
        n = dsp.make_noise_floor(ref, 20000, np.random.default_rng(0))
        self.assertEqual(len(n), 20000)
        self.assertTrue(np.isfinite(n).all())
        self.assertTrue(np.isfinite(dsp.breath(4000, np.random.default_rng(0))).all())


class TestFeatures(unittest.TestCase):
    def test_shapes(self):
        x = fake_hey_google()
        f = features.frame_features(x)
        self.assertEqual(f.shape[0], features.n_frames(len(x)))
        self.assertTrue(np.isfinite(f).all())
        self.assertEqual(features.frame_features(np.zeros(100, np.float32)).shape[0], 0)


def posteriors(seq):
    lab = []
    for c, n in seq:
        lab += [c] * n
    P = np.full((len(lab), 5), 0.01)
    P[np.arange(len(lab)), lab] = 0.96
    return np.log(P / P.sum(1, keepdims=True)).astype(np.float32)


class TestDecoder(unittest.TestCase):
    """The grammar must accept stuttered forms of the phrase and reject everything else."""
    O = 4

    def deficit(self, seq):
        return decoder.decode(posteriors(seq))["deficit"]

    def test_accepts_fluent_and_disfluent(self):
        ok = {"fluent": [(BG, 30), (HEY, 25), (GOO, 30), (GLE, 25), (BG, 30)],
              "repetition": [(BG, 30), (HEY, 8), (BG, 10), (HEY, 8), (BG, 10), (HEY, 25), (GOO, 30), (GLE, 25), (BG, 20)],
              "prolongation": [(BG, 30), (HEY, 90), (GOO, 30), (GLE, 25), (BG, 20)],
              "block": [(BG, 20), (HEY, 25), (BG, 120), (GOO, 30), (GLE, 25), (BG, 20)],
              "goo-goo": [(BG, 20), (HEY, 25), (GOO, 6), (BG, 8), (GOO, 6), (BG, 8), (GOO, 30), (GLE, 25), (BG, 20)]}
        for name, seq in ok.items():
            self.assertLess(self.deficit(seq), 8.0, name)

    def test_rejects_non_wake_word(self):
        bad = {"hey only": [(BG, 30), (HEY, 25), (BG, 40)],
               "google only": [(BG, 30), (GOO, 30), (GLE, 25), (BG, 30)],
               "reversed": [(BG, 10), (GOO, 30), (GLE, 25), (BG, 10), (HEY, 25), (BG, 10)],
               "speech": [(BG, 10), (self.O, 100), (BG, 10)],
               "silence": [(BG, 150)]}
        for name, seq in bad.items():
            self.assertGreater(self.deficit(seq), 12.0, name)


class TestDecisionRules(unittest.TestCase):
    """v1.2: every unit must be heard; fillers and restarts are tolerated; disfluency is never rewarded."""

    def test_unit_gate_rejects_partial_phrases(self):
        O = 4
        bad = {"stuttered hey only": [(BG, 30), (HEY, 8), (BG, 10), (HEY, 8), (BG, 10), (HEY, 40), (BG, 60)],
               "hey + other word": [(BG, 30), (HEY, 25), (BG, 10), (O, 40), (BG, 30)],
               "other word + google": [(BG, 30), (O, 30), (BG, 10), (GOO, 30), (GLE, 25), (BG, 30)]}
        for name, seq in bad.items():
            d = decoder.decode(posteriors(seq))
            self.assertIsNone(d["features"], name)
            self.assertTrue(d.get("gated"), name)

    def test_accepts_interjection_and_revision(self):
        O = 4
        ok = {"fluent short": [(BG, 20), (HEY, 12), (GOO, 14), (GLE, 10), (BG, 20)],
              "hey um google": [(BG, 20), (HEY, 25), (BG, 8), (O, 35), (BG, 8), (GOO, 30), (GLE, 25), (BG, 20)],
              "hey goo- hey google": [(BG, 20), (HEY, 25), (BG, 5), (GOO, 15), (BG, 40), (HEY, 25), (GOO, 30), (GLE, 25), (BG, 20)]}
        for name, seq in ok.items():
            d = decoder.decode(posteriors(seq))
            self.assertIsNotNone(d["features"], name)
            self.assertLess(d["deficit"], 12.0, name)

    def test_verifier_ignores_length_features(self):
        from wakeword.compare import VERIFIER_FEATURES
        for f in ("span_s", "n_hey", "n_goo", "n_gle", "restarts_hey", "restarts_goo", "gap_total_s", "deficit_total"):
            self.assertNotIn(f, VERIFIER_FEATURES)


class TestSynthesis(unittest.TestCase):
    def setUp(self):
        x = fake_hey_google()
        self.u = build_utterance(x)
        self.assertIsNotNone(self.u)

    def test_every_type_builds_clean_audio_and_labels(self):
        rng = np.random.default_rng(1)
        for t in ["fluent"] + DISFLUENCY_TYPES:
            for _ in range(3):
                s = DisfluencySynthesizer(self.u, rng).build(random_spec(rng, t))
                self.assertTrue(np.isfinite(s.audio).all(), t)
                lab = s.frame_labels()
                self.assertEqual(len(lab), features.n_frames(len(s.audio)))
                for c in (HEY, GOO, GLE):
                    self.assertTrue((lab == c).any(), f"{t}: class {c} missing")
                if t != "fluent":
                    self.assertTrue(s.events, t)

    def test_disfluent_is_longer_than_fluent(self):
        rng = np.random.default_rng(2)
        flu = DisfluencySynthesizer(self.u, rng).build({"type": "fluent"})
        lens = [len(DisfluencySynthesizer(self.u, rng).build(random_spec(rng, "sound_prolongation")).audio) for _ in range(5)]
        self.assertGreater(np.mean(lens), len(flu.audio))

    def test_negatives(self):
        rng = np.random.default_rng(3)
        other = [fake_hey_google(5)]
        for k in HARD_KINDS:
            s = hard_negative(self.u, k, rng, other)
            self.assertTrue(np.isfinite(s.audio).all(), k)
            self.assertFalse(s.label == "fluent")
        for k in NOISE_KINDS:
            x, lab = noise_negative(k, rng, self.u.noise)
            self.assertTrue(np.isfinite(x).all(), k)
            self.assertEqual(len(lab), features.n_frames(len(x)))


if __name__ == "__main__":
    unittest.main()

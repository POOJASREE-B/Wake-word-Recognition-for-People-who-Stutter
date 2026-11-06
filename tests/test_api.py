"""API tests. They use the trained model if present; otherwise the model-dependent ones are skipped."""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                    # noqa: E402

from server.app import create_app                     # noqa: E402
from wakeword import audio_io                         # noqa: E402
from tests.test_core import fake_hey_google           # noqa: E402

HAVE_MODEL = (ROOT / "models" / "meta.json").exists()


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["WW_RESEARCH"] = "1"                 # the research views are tested here; public mode is tested below
        try:
            cls.c = create_app().test_client()
        finally:
            os.environ.pop("WW_RESEARCH", None)
        cls.pub = create_app().test_client()

    def test_public_mode_blocks_research_views(self):
        self.assertFalse(self.pub.get("/api/health").get_json()["research"])
        for path in ("/api/dataset/summary", "/api/dataset/audio/x", "/api/live/eval", "/api/live/patterns", "/api/model/comparison"):
            self.assertEqual(self.pub.get(path).status_code, 404, path)
        self.assertEqual(self.pub.post("/api/synthesize", json={}).status_code, 404)
        self.assertEqual(self.pub.get("/").status_code, 200)

    def test_frontend_is_served(self):
        self.assertEqual(self.c.get("/").status_code, 200)
        self.assertEqual(self.c.get("/static/app.js").status_code, 200)
        self.assertEqual(self.c.get("/static/style.css").status_code, 200)

    def test_health(self):
        j = self.c.get("/api/health").get_json()
        self.assertEqual(j["status"], "ok")

    def test_garbage_audio_is_a_clean_400(self):
        r = self.c.post("/api/detect", data=b"definitely not audio", content_type="audio/wav")
        self.assertIn(r.status_code, (400, 503))
        self.assertIn("error", r.get_json())

    @unittest.skipUnless(HAVE_MODEL, "no trained model yet")
    def test_detect_returns_structured_result(self):
        wav = audio_io.wav_bytes(fake_hey_google())
        j = self.c.post("/api/detect", data=wav, content_type="audio/wav").get_json()
        for k in ("detected", "score", "threshold", "fluent_output", "events", "units"):
            self.assertIn(k, j)

    @unittest.skipUnless(HAVE_MODEL, "no trained model yet")
    def test_silence_and_noise_are_not_the_wake_word(self):
        rng = np.random.default_rng(0)
        for x in (np.zeros(SR_ := 32000, np.float32), (rng.standard_normal(SR_) * 0.05).astype(np.float32)):
            j = self.c.post("/api/detect", data=audio_io.wav_bytes(x), content_type="audio/wav").get_json()
            self.assertFalse(j["detected"])

    def test_synthesize_endpoint(self):
        import io
        wav = audio_io.wav_bytes(fake_hey_google())
        r = self.c.post("/api/synthesize", data={"audio": (io.BytesIO(wav), "a.wav"), "n": "1"}, content_type="multipart/form-data")
        self.assertIn(r.status_code, (200, 422))
        if r.status_code == 200:
            self.assertGreaterEqual(len(r.get_json()["items"]), 2)


if __name__ == "__main__":
    unittest.main()

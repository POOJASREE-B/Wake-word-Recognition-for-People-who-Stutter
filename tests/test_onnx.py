"""The ONNX frame models must give the same log-posteriors as the PyTorch checkpoints they were exported from."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                    # noqa: E402

from wakeword import onnx_models                      # noqa: E402

MODELS = ROOT / "models"
HAVE_ONNX = onnx_models.available(MODELS)
HAVE_TORCH = importlib.util.find_spec("torch") is not None and (MODELS / "frame_model.pt").exists()


@unittest.skipUnless(HAVE_ONNX, "no ONNX frame models in models/")
class TestOnnx(unittest.TestCase):
    def test_detector_uses_onnx(self):
        from wakeword.detector import WakeWordDetector
        det = WakeWordDetector.load(MODELS)
        self.assertEqual(type(det.model).__name__, "OnnxEnsembleModel")

    @unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
    def test_matches_pytorch(self):
        from wakeword.embtrain import EnsembleFrameModel
        torch_model = EnsembleFrameModel.load(MODELS / "frame_model.pt")
        onnx_model = onnx_models.OnnxEnsembleModel.load(MODELS)
        rng = np.random.default_rng(3)
        for sec in (0.6, 2.5, 6.0):
            x = (rng.standard_normal(int(16000 * sec)) * 0.05).astype(np.float32)
            diff = np.abs(torch_model.predict_wave(x) - onnx_model.predict_wave(x)).max()
            self.assertLess(diff, 1e-4, "re-run scripts/export_onnx.py after retraining")


if __name__ == "__main__":
    unittest.main()

"""ONNX Runtime versions of the two frame models, so the web app runs without PyTorch.

`scripts/export_onnx.py` writes models/frame_model.onnx (PCEN CRNN) and models/frame_emb.onnx (embedding head) from the
PyTorch checkpoints. Each graph includes the input normalisation and the final log-softmax, so the classes below only
compute the features, run the graph and combine the results exactly as the PyTorch classes do. The detector uses
these files when they exist (see detector.load_frame_model); training still needs PyTorch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import CLASSES
from .features import n_frames, pcen

ONNX_FILES = ("frame_model.onnx", "frame_emb.onnx")


def _session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1          # one request at a time per thread; Waitress already serves requests in parallel
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


class OnnxCrnnModel:
    """Same output as torch_models.TorchFrameModel: predict_wave(x) -> (T, 5) log-posteriors at 10 ms frames."""
    arch = "crnn"

    def __init__(self, path):
        self.sess = _session(path)
        self.inp = self.sess.get_inputs()[0].name

    def predict_wave(self, x: np.ndarray) -> np.ndarray:
        f = pcen(x)
        if f.shape[0] == 0:
            return np.zeros((0, len(CLASSES)), dtype=np.float32)
        return self.sess.run(None, {self.inp: f[None].astype(np.float32)})[0][0].astype(np.float32)


class OnnxEmbModel:
    """Same output as embtrain.EmbFrameModel."""
    arch = "emb"

    def __init__(self, path):
        self.sess = _session(path)
        self.inp = self.sess.get_inputs()[0].name

    def head_logp(self, e):
        if len(e) == 0:
            return np.zeros((0, len(CLASSES)), np.float32)
        return self.sess.run(None, {self.inp: np.asarray(e, np.float32)[None]})[0][0]

    def predict_wave(self, x):
        from .embtrain import _upsample, embed_cached
        return _upsample(self.head_logp(embed_cached(x)), n_frames(len(x)))


class OnnxEnsembleModel:
    """Posterior average of the two members, computed as in embtrain.EnsembleFrameModel."""
    arch = "ensemble"

    def __init__(self, models, weights=None):
        self.models = list(models)
        self.w = np.asarray(weights if weights is not None else [1.0] * len(models), np.float64)
        self.w = self.w / self.w.sum()

    def predict_wave(self, x):
        lps = [m.predict_wave(x) for m in self.models]
        T = min(len(l) for l in lps)
        st = np.stack([l[:T] for l in lps])
        mx = st.max(0)
        out = mx + np.log(np.tensordot(self.w, np.exp(st - mx), axes=1))
        return out.astype(np.float32)

    @classmethod
    def load(cls, model_dir):
        d = Path(model_dir)
        return cls([OnnxCrnnModel(d / ONNX_FILES[0]), OnnxEmbModel(d / ONNX_FILES[1])])


def available(model_dir) -> bool:
    d = Path(model_dir)
    return all((d / n).exists() for n in ONNX_FILES)

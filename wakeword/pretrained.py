"""Frozen pretrained speech embedding (Google speech-embedding as shipped with openWakeWord, Apache-2.0).

wave -> 32-band mel (10 ms hop, ONNX) -> 76-frame windows -> 96-d embedding every `stride` mel frames.
The embeddings are the input of a small trainable head (EmbFrameModel) with the SAME interface as the CNN/CRNN
frame models, so the decoder, verifier, livesim and the page work unchanged."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .config import CLASSES, SR
from .features import n_frames

WIN_MEL, EMB = 76, 96
_SESS = {}
URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/{}.onnx"


def _path(name):
    for d in (os.environ.get("WW_PRETRAINED", ""), str(Path(__file__).resolve().parent.parent / "pretrained")):
        if d and (Path(d) / f"{name}.onnx").exists():
            return str(Path(d) / f"{name}.onnx")
    d = Path(__file__).resolve().parent.parent / "pretrained"
    d.mkdir(exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(URL.format(name), str(d / f"{name}.onnx"))
    return str(d / f"{name}.onnx")


def _sess(name):
    if name not in _SESS:
        import onnxruntime as ort
        o = ort.SessionOptions(); o.intra_op_num_threads = 2
        _SESS[name] = ort.InferenceSession(_path(name), o, providers=["CPUExecutionProvider"])
    return _SESS[name]


def mel(x: np.ndarray) -> np.ndarray:
    """(n,) float waveform in [-1, 1] -> (frames, 32) log-mel (the model's own scaling)."""
    s = _sess("melspectrogram")
    y = np.clip(np.asarray(x, np.float32), -1, 1) * 32767.0
    m = s.run(None, {"input": y[None]})[0]
    m = np.squeeze(m)
    return (m / 10.0 + 2.0).astype(np.float32)


def embed(x: np.ndarray, stride: int = 4) -> np.ndarray:
    """(n,) -> (K, 96): embedding for the window ending at mel frame 75 + k*stride (K may be 0 for short audio)."""
    x = np.asarray(x, np.float32)
    pad = np.zeros(SR // 2, np.float32)                    # half a second of context on both sides so short audio works
    m = mel(np.concatenate([pad, x, pad]))
    T = len(m)
    if T < WIN_MEL:
        return np.zeros((0, EMB), np.float32)
    starts = np.arange(0, T - WIN_MEL + 1, stride)
    w = np.stack([m[s:s + WIN_MEL] for s in starts])[..., None]
    out = []
    s = _sess("embedding_model")
    for i in range(0, len(w), 256):
        out.append(np.squeeze(s.run(None, {"input_1": w[i:i + 256]}), axis=(1, 2)) if False else s.run(None, {"input_1": w[i:i + 256]})[0].reshape(-1, EMB))
    return np.concatenate(out).astype(np.float32)


def centers(n_emb: int, stride: int = 4):
    """time (in 10 ms frames of the ORIGINAL audio) at the centre of each embedding window."""
    # window k covers padded mel frames [k*stride, k*stride+76); pad = 50 frames
    return np.arange(n_emb) * stride + WIN_MEL // 2 - 50

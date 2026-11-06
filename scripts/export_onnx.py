"""Export the installed PyTorch frame models to ONNX so the web app can run without PyTorch.

    python scripts/export_onnx.py            (needs PyTorch; run it again after every retraining)

Writes models/frame_model.onnx and models/frame_emb.onnx. Each graph contains the input normalisation, the network and
the final log-softmax. The script then compares ONNX Runtime with PyTorch on generated audio and stops with an error if
the log-posteriors differ by more than 1e-4.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402

from wakeword.config import MODEL_DIR, N_MELS                        # noqa: E402
from wakeword import pretrained as P                                 # noqa: E402
from wakeword.embtrain import EmbFrameModel                          # noqa: E402
from wakeword.torch_models import TorchFrameModel                    # noqa: E402
from wakeword.onnx_models import OnnxCrnnModel, OnnxEmbModel, ONNX_FILES   # noqa: E402


class Wrapped(torch.nn.Module):
    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.as_tensor(np.asarray(mean, np.float32)))
        self.register_buffer("std", torch.as_tensor(np.asarray(std, np.float32)))

    def forward(self, x):
        return torch.log_softmax(self.net((x - self.mean) / self.std), dim=-1)


def export(model, n_in, out_path):
    w = Wrapped(model.net.eval(), model.mean, model.std).eval()
    dummy = torch.randn(1, 120, n_in)
    torch.onnx.export(w, dummy, str(out_path), input_names=["x"], output_names=["logp"],
                      dynamic_axes={"x": {1: "frames"}, "logp": {1: "frames"}}, opset_version=17, dynamo=False)


def main(model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    crnn = TorchFrameModel.load(model_dir / "frame_model.pt")
    emb = EmbFrameModel.load(model_dir / "frame_emb.pt")
    export(crnn, N_MELS, model_dir / ONNX_FILES[0])
    export(emb, P.EMB, model_dir / ONNX_FILES[1])

    rng = np.random.default_rng(0)
    worst = 0.0
    for sec in (0.5, 1.7, 3.0, 6.0):
        x = (rng.standard_normal(int(16000 * sec)) * 0.05).astype(np.float32)
        for t_model, o_model in ((crnn, OnnxCrnnModel(model_dir / ONNX_FILES[0])), (emb, OnnxEmbModel(model_dir / ONNX_FILES[1]))):
            a, b = t_model.predict_wave(x), o_model.predict_wave(x)
            worst = max(worst, float(np.abs(a - b).max()))
    print(f"largest difference between PyTorch and ONNX log-posteriors: {worst:.2e}")
    if worst > 1e-4:
        sys.exit("ONNX export does not match PyTorch; the .onnx files should not be used")
    print("wrote", *(model_dir / n for n in ONNX_FILES))


if __name__ == "__main__":
    main(*sys.argv[1:])

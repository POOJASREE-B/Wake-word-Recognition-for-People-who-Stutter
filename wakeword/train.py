"""Training helpers shared by the pipeline and the model comparison."""
from __future__ import annotations

from .config import DATA_DIR, MODEL_DIR
from .model import load_sample, read_annotations, read_manifest
from . import decoder


def decode_split(model, data_dir, rows, ann, log=print, gate="default"):
    """Run the grammar decoder over every clip -> [(row, decode_result)].  gate=None decodes without the unit gate."""
    out = []
    for i, r in enumerate(rows):
        x, _ = load_sample(data_dir, r, ann)
        lp = model.predict_wave(x)
        out.append((r, decoder.decode(lp) if gate == "default" else decoder.decode(lp, gate)))
        if (i + 1) % 500 == 0:
            log(f"      decoded {i + 1}/{len(rows)}")
    return out


def train_all(data_dir=DATA_DIR, model_dir=MODEL_DIR, epochs: int = 12, names=("mlp_small",), log=print, **kw):
    """Train + select among the given candidates (default: one fast scikit-learn model)."""
    from .compare import run_comparison
    det, _ = run_comparison(data_dir, model_dir, names=list(names), epochs_mlp=epochs, log=log, **kw)
    return det

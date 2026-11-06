"""End-to-end pipeline:  build -> quick model -> verify new recordings + forced alignment -> final build -> compare -> select."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from . import dataset_builder, sources
from .compare import run_comparison
from .config import DATA_DIR, MODEL_DIR, RAW_DIR
from .detector import WakeWordDetector


def run_all(raw_dir=RAW_DIR, data_dir=DATA_DIR, model_dir=MODEL_DIR, names=None, fast: bool = False,
            epochs_mlp: int = 12, epochs_torch: int = 30, sep28k_root=None, torch_max_clips=None, n_sep: int = 0,
            pos_scale: float = 1.0, pos_share: float = 0.20, target_far: float = 0.01, n_piper: int = 1000,
            bootstrap_model=None, init_from=None, lr=None, log=print):
    from .bootstrap import verify_new_recordings
    t0 = time.time()
    raw_dir, data_dir, model_dir = Path(raw_dir), Path(data_dir), Path(model_dir)
    scale = 0.35 if fast else 1.0
    stage = model_dir.parent / "_stage1"
    if bootstrap_model:
        # v1.2b: a quick first-pass model is a poor aligner (it mis-placed the Hey|Goo|gle boundaries of the real
        # recordings and rejected 40 of 75 of them); an already trained model does all three jobs much better.
        log(f"=== STAGE 1: skipped - using the trained model in {bootstrap_model} to verify and align the recordings ===")
        det1 = WakeWordDetector.load(bootstrap_model)
    else:
        det1 = None
    if det1 is None:
        log("=== STAGE 1: first-pass dataset + quick model (used to align and to verify the raw recordings) ===")
    # the first pass only has to be good enough to align and verify the recordings, so it stays small
        dataset_builder.build_dataset(stage / "data", raw_dir, scale=scale, sep28k_root=sep28k_root, n_sep=1500 if n_sep <= 0 else min(n_sep, 1500),
                                      pos_scale=min(pos_scale, 0.5), pos_share=0.40, n_piper=min(n_piper, 300), log=log)
        run_comparison(stage / "data", stage / "model", names=["mlp_small"], epochs_mlp=6 if fast else 8, evaluate_test=False, force=True, log=log)
        det1 = WakeWordDetector.load(stage / "model")

    log("=== STAGE 2: verify the raw recordings, align boundaries, build the final dataset ===")
    _, referenced = sources.load_curated(raw_dir)
    extra, audit = verify_new_recordings(det1, raw_dir, referenced, log=log)
    dataset_builder.build_dataset(data_dir, raw_dir, scale=scale, extra_sources=extra, refiner=det1.model,
                                  extra_audit=audit, sep28k_root=sep28k_root, n_sep=n_sep,
                                  pos_scale=pos_scale, pos_share=pos_share, n_piper=n_piper, log=log)
    if dataset_builder.audit_dataset(data_dir, log=log):
        raise SystemExit("The dataset audit found problems (listed above). Training was not started.")

    log("=== STAGE 3: train and compare the candidate models, keep the best ===")
    det, out = run_comparison(data_dir, model_dir, names=names, epochs_mlp=6 if fast else epochs_mlp,
                              epochs_torch=5 if fast else epochs_torch, force=fast, torch_max_clips=torch_max_clips,
                              target_far=target_far, init_from=init_from, lr=lr, log=log)
    shutil.rmtree(stage, ignore_errors=True)
    log(f"=== finished in {(time.time() - t0) / 60:.1f} min - selected model: {out['selected']} ===")
    return det, out

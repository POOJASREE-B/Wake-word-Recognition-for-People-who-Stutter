"""Optional PyTorch frame models (temporal CNN and CNN + BiGRU).

These plug into exactly the same decoder / verifier as the scikit-learn MLP, so the model
comparison (wakeword/compare.py) can rank all of them on the same held-out data.
If PyTorch is not installed this module still imports; the candidates are simply skipped.
"""
from __future__ import annotations

import time

import numpy as np

from .config import CLASSES, IGNORE, N_MELS
from .features import pcen

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:                                    # pragma: no cover - depends on the machine
    torch, nn, HAVE_TORCH = None, None, False

ARCHS = ("cnn", "crnn")


def make_net(arch: str, n_mels: int = N_MELS, n_cls: int = len(CLASSES), ch: int = 64, hid: int = 96):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch = arch
            self.front = nn.Sequential(
                nn.Conv1d(n_mels, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(),
                nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(0.1))
            if arch == "crnn":
                self.rnn = nn.GRU(ch, hid, batch_first=True, bidirectional=True)
                out = 2 * hid
            else:                                    # dilated temporal convolutions (receptive field ~ +-0.4 s)
                self.tcn = nn.Sequential(
                    nn.Conv1d(ch, ch, 3, padding=2, dilation=2), nn.BatchNorm1d(ch), nn.ReLU(),
                    nn.Conv1d(ch, ch, 3, padding=4, dilation=4), nn.BatchNorm1d(ch), nn.ReLU(),
                    nn.Conv1d(ch, ch, 3, padding=8, dilation=8), nn.BatchNorm1d(ch), nn.ReLU(),
                    nn.Conv1d(ch, ch, 3, padding=16, dilation=16), nn.BatchNorm1d(ch), nn.ReLU())
                out = ch
            self.head = nn.Linear(out, n_cls)

        def forward(self, x):                        # x: (B, T, n_mels)
            h = self.front(x.transpose(1, 2))        # (B, ch, T)
            if self.arch == "crnn":
                h, _ = self.rnn(h.transpose(1, 2))   # (B, T, 2*hid)
            else:
                h = self.tcn(h).transpose(1, 2)      # (B, T, ch)
            return self.head(h)                      # (B, T, n_cls)
    return Net()


class TorchFrameModel:
    """Same interface as model.FrameModel: predict_wave(x) -> (T, 5) log-posteriors."""

    def __init__(self, net, mean, std, arch):
        self.net, self.arch = net, arch
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.device = "cpu"
        self.net.eval()

    def predict_wave(self, x: np.ndarray) -> np.ndarray:
        f = pcen(x)
        if f.shape[0] == 0:
            return np.zeros((0, len(CLASSES)), dtype=np.float32)
        t = torch.from_numpy(((f - self.mean) / self.std).astype(np.float32))[None]
        with torch.no_grad():
            lp = torch.log_softmax(self.net(t), dim=-1)[0]
        return lp.numpy().astype(np.float32)

    def save(self, path):
        torch.save({"arch": self.arch, "state": self.net.state_dict(),
                    "mean": torch.from_numpy(self.mean), "std": torch.from_numpy(self.std)}, str(path))

    @classmethod
    def load(cls, path):
        try:
            d = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:                            # old PyTorch without the weights_only argument
            d = torch.load(str(path), map_location="cpu")
        net = make_net(d["arch"])
        net.load_state_dict(d["state"])
        return cls(net, d["mean"].numpy(), d["std"].numpy(), d["arch"])


def _load_clips(data_dir, rows, ann):
    from .model import load_sample
    feats, labs = [], []
    for r in rows:
        x, lab = load_sample(data_dir, r, ann)
        f = pcen(x)
        n = min(len(f), len(lab))
        lab = lab[:n].copy()
        lab[lab == IGNORE] = -100
        feats.append(f[:n])
        labs.append(lab)
    return feats, labs


def _batch(feats, labs, idx, mean, std, freq_mask=False, rng=None):
    T = max(len(feats[i]) for i in idx)
    X = np.zeros((len(idx), T, N_MELS), dtype=np.float32)
    Y = np.full((len(idx), T), -100, dtype=np.int64)
    for k, i in enumerate(idx):
        f = (feats[i] - mean) / std
        if freq_mask and rng is not None and rng.random() < 0.5:
            w = int(rng.integers(1, 4))
            s = int(rng.integers(0, N_MELS - w))
            f = f.copy()
            f[:, s:s + w] = 0.0
        X[k, :len(f)] = f
        Y[k, :len(f)] = labs[i]
    return torch.from_numpy(X), torch.from_numpy(Y)


def train_torch(arch: str, data_dir, out_path, epochs: int = 30, seed: int = 0, log=print, max_train_clips=None,
                init_from=None, lr=None):
    """Train a PyTorch frame model on the clips of the 'train' split, select the epoch on 'val'."""
    from .model import read_annotations, read_manifest
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rows, ann = read_manifest(data_dir), read_annotations(data_dir)
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    if max_train_clips:
        tr = [tr[i] for i in rng.permutation(len(tr))[:max_train_clips]]
    log(f"[{arch}] loading {len(tr)} train / {len(va)} val clips ...")
    ftr, ltr = _load_clips(data_dir, tr, ann)
    fva, lva = _load_clips(data_dir, va, ann)
    init = None
    if init_from is not None:
        try:
            init = TorchFrameModel.load(init_from)
            if init.arch != arch:
                log(f"[{arch}] not fine-tuning: {init_from} holds a '{init.arch}' model")
                init = None
        except Exception as exc:
            log(f"[{arch}] not fine-tuning: cannot load {init_from} ({exc})")
            init = None
    if init is not None:            # fine-tuning keeps the feature normalisation the weights were trained with
        mean, std = init.mean.astype(np.float32), init.std.astype(np.float32)
        log(f"[{arch}] fine-tuning from {init_from}")
    else:
        allf = np.concatenate(ftr)
        mean, std = allf.mean(axis=0).astype(np.float32), (allf.std(axis=0) + 1e-3).astype(np.float32)
    cnt = np.bincount(np.concatenate([l[l >= 0] for l in ltr]), minlength=len(CLASSES)).astype(np.float64)
    w = (cnt.sum() / np.maximum(cnt, 1)) ** 0.5
    w = (w / w.mean()).astype(np.float32)
    net = make_net(arch)
    if init is not None:
        net.load_state_dict(init.net.state_dict())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(dev)
    lr = lr or (5e-4 if init is not None else 2e-3)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    lossf = nn.CrossEntropyLoss(weight=torch.from_numpy(w).to(dev), ignore_index=-100)
    best, best_state, t0 = -1.0, None, time.time()
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(ftr))
        total = 0.0
        for s in range(0, len(order), 16):
            idx = order[s:s + 16]
            X, Y = _batch(ftr, ltr, idx, mean, std, freq_mask=True, rng=rng)
            X, Y = X.to(dev), Y.to(dev)
            opt.zero_grad()
            loss = lossf(net(X).reshape(-1, len(CLASSES)), Y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(idx)
        sched.step()
        net.eval()
        hit, tot = np.zeros(len(CLASSES)), np.zeros(len(CLASSES))
        with torch.no_grad():
            for s in range(0, len(fva), 32):
                idx = list(range(s, min(s + 32, len(fva))))
                X, Y = _batch(fva, lva, idx, mean, std)
                pred = net(X.to(dev)).argmax(-1).cpu().numpy()
                Yn = Y.numpy()
                for c in range(len(CLASSES)):
                    m = Yn == c
                    tot[c] += m.sum()
                    hit[c] += (pred[m] == c).sum()
        rec = float(np.mean(hit[tot > 0] / tot[tot > 0]))
        log(f"[{arch}] epoch {ep + 1:2d}/{epochs}  loss={total / len(order):.4f}  val macro-recall={rec:.3f}  ({time.time() - t0:.0f}s)")
        if rec > best:
            best = rec
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.cpu()
    model = TorchFrameModel(net, mean, std, arch)
    if out_path:
        model.save(out_path)
    return model, {"val_macro_recall": round(best, 4)}


def selftest(log=print) -> bool:
    """One tiny forward/backward pass per architecture: `python run.py selftest-torch`."""
    if not HAVE_TORCH:
        log("PyTorch is not installed -> torch candidates will be skipped (pip install torch).")
        return False
    for arch in ARCHS:
        net = make_net(arch)
        x = torch.randn(2, 120, N_MELS)
        y = torch.randint(0, len(CLASSES), (2, 120))
        loss = nn.CrossEntropyLoss()(net(x).reshape(-1, len(CLASSES)), y.reshape(-1))
        loss.backward()
        m = TorchFrameModel(net, np.zeros(N_MELS), np.ones(N_MELS), arch)
        lp = m.predict_wave(np.random.randn(16000).astype(np.float32) * 0.05)
        assert lp.shape[1] == len(CLASSES) and np.isfinite(lp).all()
        log(f"  {arch}: forward/backward/inference OK (output {lp.shape})")
    return True

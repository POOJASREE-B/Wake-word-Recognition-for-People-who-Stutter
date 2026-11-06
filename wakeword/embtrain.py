"""Frame model on top of a FROZEN pretrained speech embedding (see pretrained.py).

EmbFrameModel has the interface of TorchFrameModel (predict_wave -> (T, 5) log-posteriors at 10 ms), so the decoder,
verifier, livesim, evaluation and the page work unchanged.  Only a small head (Linear + BiGRU + Linear, ~0.1 M
parameters) is trained, which is what makes it practical with 23 real speakers and no GPU."""
from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path

import numpy as np

from . import pretrained as P
from .config import BG, CLASSES, GLE, GOO, HEY, IGNORE, SR
from .features import n_frames

try:
    import torch
    import torch.nn as nn
except Exception:                                      # pragma: no cover
    torch = nn = None

STRIDE = 2                  # embedding every 20 ms
TOPK_E = 4                  # frames per unit for the clip-level loss (20 ms frames)
HID = 96
_CACHE = {}


def _key(x):
    x = np.asarray(x, np.float32)
    return hashlib.md5(x[:: max(1, len(x) // 4096)].tobytes() + str(len(x)).encode()).hexdigest()


def embed_cached(x):
    k = _key(x)
    if k not in _CACHE:
        if len(_CACHE) > 6000:
            _CACHE.clear()
        _CACHE[k] = P.embed(x, STRIDE)
    return _CACHE[k]


def make_net():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(P.EMB, HID), nn.ReLU(), nn.Dropout(0.15))
            self.rnn = nn.GRU(HID, HID, batch_first=True, bidirectional=True)
            self.head = nn.Linear(2 * HID, len(CLASSES))

        def forward(self, x):                           # (B, K, 96)
            h, _ = self.rnn(self.inp(x))
            return self.head(h)
    return Net()


def _upsample(lp, n_out):
    """(K, 5) log-posteriors at the window centres -> (n_out, 5) at 10 ms frames."""
    K = lp.shape[0]
    if K == 0:
        return np.tile(np.log(np.array([0.96, 0.01, 0.01, 0.01, 0.01], np.float32)), (n_out, 1))
    c = P.centers(K, STRIDE).astype(np.float64)
    t = np.arange(n_out, dtype=np.float64)
    out = np.stack([np.interp(t, c, lp[:, j]) for j in range(lp.shape[1])], axis=1)
    out -= np.log(np.exp(out).sum(1, keepdims=True))
    return out.astype(np.float32)


class EmbFrameModel:
    arch = "emb"

    def __init__(self, net, mean, std):
        self.net = net
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.net.eval()

    def head_logp(self, e):                              # (K, 96) -> (K, 5)
        if len(e) == 0:
            return np.zeros((0, len(CLASSES)), np.float32)
        t = torch.from_numpy(((e - self.mean) / self.std).astype(np.float32))[None]
        with torch.no_grad():
            return torch.log_softmax(self.net(t), -1)[0].numpy()

    def predict_wave(self, x):
        n = n_frames(len(x))
        return _upsample(self.head_logp(embed_cached(x)), n)

    def save(self, path):
        torch.save({"arch": "emb", "state": self.net.state_dict(), "mean": torch.from_numpy(self.mean),
                    "std": torch.from_numpy(self.std)}, str(path))

    @classmethod
    def load(cls, path):
        try:
            d = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            d = torch.load(str(path), map_location="cpu")
        net = make_net()
        net.load_state_dict(d["state"])
        return cls(net, d["mean"].numpy(), d["std"].numpy())


# ----------------------------------------------------------------------------- data
def load_replay_cache(path=None):
    """Cached embeddings of the replay set (written during training); default location: <project>/cache/emb_replay.pkl."""
    path = Path(path) if path else Path(__file__).resolve().parent.parent / "cache" / "emb_replay.pkl"
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _batch(embs, labs, idx, mean, std, rng=None, noise=0.0):
    T = max(len(embs[i]) for i in idx)
    X = np.zeros((len(idx), T, P.EMB), np.float32)
    Y = np.full((len(idx), T), -100, np.int64)
    for k, i in enumerate(idx):
        e = (embs[i].astype(np.float32) - mean) / std
        if noise and rng is not None:
            e = e + rng.normal(0, noise, e.shape).astype(np.float32)
        X[k, :len(e)] = e
        lab = labs[i].astype(np.int64).copy()
        lab[lab == IGNORE] = -100
        Y[k, :len(lab)] = lab[:len(e)]
    return torch.from_numpy(X), torch.from_numpy(Y)


def pretrain(replay_rows, cache, epochs=30, seed=0, log=print):
    """Train the head from scratch on the replay dataset's train clips; epoch chosen on its val clips."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr = [r for r in replay_rows if r["split"] == "train" and r["sample_id"] in cache]
    va = [r for r in replay_rows if r["split"] == "val" and r["sample_id"] in cache]
    E = [cache[r["sample_id"]][0] for r in tr]
    L = [cache[r["sample_id"]][1] for r in tr]
    Ev = [cache[r["sample_id"]][0] for r in va[:700]]
    Lv = [cache[r["sample_id"]][1] for r in va[:700]]
    allf = np.concatenate([e.astype(np.float32) for e in E])
    mean, std = allf.mean(0), allf.std(0) + 1e-3
    cnt = np.bincount(np.concatenate([l[(l >= 0) & (l < len(CLASSES))] for l in L]).astype(np.int64), minlength=len(CLASSES)).astype(np.float64)
    w = (cnt.sum() / np.maximum(cnt, 1)) ** 0.5
    w = torch.from_numpy((w / w.mean()).astype(np.float32))
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(weight=w, ignore_index=-100)
    best, bs, t0 = -1, None, time.time()
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(E)); tot = 0.0
        for s in range(0, len(order), 16):
            X, Y = _batch(E, L, order[s:s + 16], mean, std, rng, noise=0.05)
            opt.zero_grad()
            loss = lossf(net(X).reshape(-1, len(CLASSES)), Y.reshape(-1))
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            tot += float(loss) * len(X)
        sched.step(); net.eval()
        hit, tt = np.zeros(len(CLASSES)), np.zeros(len(CLASSES))
        with torch.no_grad():
            for s in range(0, len(Ev), 32):
                X, Y = _batch(Ev, Lv, list(range(s, min(s + 32, len(Ev)))), mean, std)
                pr = net(X).argmax(-1).numpy(); Yn = Y.numpy()
                for c in range(len(CLASSES)):
                    m = Yn == c; tt[c] += m.sum(); hit[c] += (pr[m] == c).sum()
        rec = float(np.mean(hit[tt > 0] / tt[tt > 0]))
        log(f"[emb] epoch {ep + 1:2d}/{epochs} loss={tot / len(E):.4f} val macro-recall={rec:.3f} ({time.time() - t0:.0f}s)")
        if rec > best:
            best, bs = rec, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(bs)
    return EmbFrameModel(net, mean, std)


# ----------------------------------------------------------------------------- real clips: voice + simulated disfluency + channel
def _variant_wave(x, rng, p_disfl=0.35):
    from .augment import augment_channel
    from .robust import disfluent, modulate
    y = x
    if rng.random() < p_disfl:
        y = disfluent(y, rng)
    y = modulate(y, rng)
    y, _ = augment_channel(y, rng, strength=1.0)
    return y.astype(np.float32)


def real_variants_emb(items, n_aug, seed=0, p_disfl=0.35, cache_path=None):
    """[(emb (K,96), frame labels (K,) or None, wake, item)] ; same clip -> same variants every call (cached)."""
    from . import realpeople as rp
    key = (n_aug, seed, p_disfl)
    store = {}
    if cache_path and Path(cache_path).exists():
        store = pickle.load(open(cache_path, "rb"))
    out, dirty = [], False
    for it in items:
        k = (it.id,) + key
        if k not in store:
            rng = np.random.default_rng(int(hashlib.md5(f"{it.id}{seed}".encode()).hexdigest()[:8], 16))
            vs = []
            for j in range(n_aug + 1):
                x = it.x if j == 0 else _variant_wave(it.x, rng, p_disfl)
                e = P.embed(x, STRIDE)
                lab = None
                if not it.wake:
                    l10 = rp.frame_labels_negative(x, it.phrase)
                    c = np.clip(P.centers(len(e), STRIDE), 0, max(len(l10) - 1, 0))
                    lab = l10[c].astype(np.int64)
                vs.append((e.astype(np.float16), lab))
            store[k] = vs; dirty = True
        out += [(e, l, it.wake, it) for e, l in store[k]]
    if cache_path and dirty:
        pickle.dump(store, open(cache_path, "wb"))
    return out


def _topk(lp, c, k=TOPK_E):
    return torch.topk(lp[:, c], min(k, lp.shape[0])).values.mean()


def finetune_emb(model, replay, real_items, epochs=10, per_epoch=480, lr=1e-3, seed=0, log=print, n_aug=14,
                 var_cache=None, rep_cache=None, mil_weight=1.0, **_):
    """Same role as realtrain.finetune: replay clips + real variants (positives: weak clip-level loss)."""
    import copy
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = copy.deepcopy(model.net)
    mean, std = model.mean, model.std
    var = real_variants_emb(real_items, n_aug, cache_path=var_cache)
    pos = [v for v in var if v[2]]
    neg = [v for v in var if not v[2]]
    cache = rep_cache
    rows = [r for r in replay["train_rows"] if r["sample_id"] in cache]
    E = [cache[r["sample_id"]][0] for r in rows]
    L = [cache[r["sample_id"]][1] for r in rows]
    w = torch.tensor([0.6, 1.5, 1.5, 1.5, 0.9], dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w, ignore_index=-100)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    t0 = time.time()
    log(f"      emb fine-tune data: {len(pos)} positive / {len(neg)} negative real variants")
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(E))[:per_epoch]
        tot, nb = 0.0, 0
        for b in range(max(1, len(order) // 12)):
            idx = order[b * 12:(b + 1) * 12]
            X, Y = _batch(E, L, idx, mean, std, rng, noise=0.05)
            opt.zero_grad()
            loss = lossf(net(X).reshape(-1, len(CLASSES)), Y.reshape(-1))
            if neg:
                sel = [int(i) for i in rng.integers(0, len(neg), 3)]
                Xn, Yn = _batch([neg[i][0] for i in sel], [neg[i][1] for i in sel], list(range(len(sel))), mean, std, rng, 0.05)
                loss = loss + lossf(net(Xn).reshape(-1, len(CLASSES)), Yn.reshape(-1))
            if pos and mil_weight > 0:
                m = 0.0
                sel = [pos[int(i)] for i in rng.integers(0, len(pos), 4)]
                for e, _, _, _ in sel:
                    xp = torch.from_numpy(((e.astype(np.float32) - mean) / std)).float()[None]
                    xp = xp + 0.05 * torch.randn_like(xp)
                    lp = torch.log_softmax(net(xp), -1)[0]
                    voiced = lp[:, BG] < np.log(0.5)
                    if int(voiced.sum()) >= TOPK_E:
                        lp = lp[voiced]
                    m = m - sum(_topk(lp, c) for c in (HEY, GOO, GLE)) / 3.0
                loss = loss + mil_weight * m / len(sel) * 0.5
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        log(f"      emb fine-tune epoch {ep + 1}/{epochs} loss={tot / max(nb, 1):.3f} ({time.time() - t0:.0f}s)")
    net.eval()
    return EmbFrameModel(net, mean, std)


# ----------------------------------------------------------------------------- ensemble of the CNN/CRNN and the embedding head
class EnsembleFrameModel:
    """Posterior-averaging ensemble of frame models (e.g. the PCEN CRNN and the pretrained-embedding head)."""
    arch = "ensemble"

    def __init__(self, models, weights=None):
        self.models = list(models)
        self.w = np.asarray(weights if weights is not None else [1.0] * len(models), np.float64)
        self.w = self.w / self.w.sum()

    def predict_wave(self, x):
        lps = [m.predict_wave(x) for m in self.models]
        T = min(len(l) for l in lps)
        st = np.stack([l[:T] for l in lps])                         # (M, T, 5)
        mx = st.max(0)
        out = mx + np.log(np.tensordot(self.w, np.exp(st - mx), axes=1))
        return out.astype(np.float32)

    def save(self, path):
        path = Path(path)
        self.models[0].save(path)                                    # CRNN -> frame_model.pt
        self.models[1].save(path.parent / "frame_emb.pt")           # embedding head

    @classmethod
    def load(cls, path):
        from .torch_models import TorchFrameModel
        path = Path(path)
        return cls([TorchFrameModel.load(path), EmbFrameModel.load(path.parent / "frame_emb.pt")])


def tune_ensemble(model, replay, items, epochs=8, per_epoch=360, seed=0, log=print, rep_cache=None, var_cache=None, emb_epochs=10, **_):
    """Fine-tune both members on the real speakers (voice modulation + simulated disfluency) and re-ensemble."""
    from . import realtrain as rt
    crnn = rt.finetune(model.models[0], replay, items, epochs, per_epoch, seed=seed, log=log)
    emb = finetune_emb(model.models[1], replay, items, emb_epochs, 480, seed=seed, log=log, rep_cache=rep_cache, var_cache=var_cache)
    return EnsembleFrameModel([crnn, emb], model.w)

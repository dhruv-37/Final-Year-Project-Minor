"""Sequence models over firm histories, pure PyTorch and CPU-friendly.

    temp_trans  Temporal Transformer: a firm's last K fiscal years as the sequence
    temp_mamba  Temporal Mamba: the same sequence through Mamba blocks

Mamba block = selective state space model of Gu & Dao (2023), written out because mamba-ssm needs
compiled CUDA extensions. Each epoch keeps all frauds and draws fresh non-frauds, so over the epochs
the model sees most of the majority class.
"""
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from torch import nn
from torch.nn import functional as F


class Preprocessor:
    """Median-impute + quantile-normalise, fit on training rows only; returns values and a NaN mask."""

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.imp_ = SimpleImputer(strategy='median').fit(X)
        self.qt_ = QuantileTransformer(n_quantiles=500, output_distribution='normal', subsample=50_000,
                                       random_state=0).fit(self.imp_.transform(X))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        v = np.clip(self.qt_.transform(self.imp_.transform(X)), -5, 5)
        return v.astype(np.float32), np.isnan(X).astype(np.float32)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps, self.w = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * self.w / (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()


class MambaBlock(nn.Module):
    """Pre-norm residual Mamba block; input and output are (batch, length, d)."""

    def __init__(self, d, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        h = expand * d
        self.d_conv, self.d_state = d_conv, d_state
        self.norm = RMSNorm(d)
        self.inp = nn.Linear(d, 2 * h)
        self.conv = nn.Conv1d(h, h, d_conv, padding=d_conv - 1, groups=h)
        self.x_proj = nn.Linear(h, 2 * d_state + 4)
        self.dt_proj = nn.Linear(4, h)
        self.A_log = nn.Parameter(torch.log(torch.arange(1.0, d_state + 1)).repeat(h, 1))
        self.D = nn.Parameter(torch.ones(h))
        self.out = nn.Linear(h, d)
        self.drop = nn.Dropout(dropout)

    def ssm(self, u):
        """h[t] = exp(A * dt[t]) h[t-1] + (dt[t] * u[t]) B[t];  y[t] = h[t] . C[t] + D u[t]."""
        B, C, dt = self.x_proj(u).split([self.d_state, self.d_state, 4], -1)
        dt = F.softplus(self.dt_proj(dt))
        decay = torch.exp(-self.A_log.exp() * dt[..., None])
        inject = (dt * u)[..., None] * B[:, :, None, :]
        h, ys = torch.zeros_like(inject[:, 0]), []
        for t in range(u.shape[1]):
            h = decay[:, t] * h + inject[:, t]
            ys.append((h * C[:, t, None]).sum(-1))
        return torch.stack(ys, 1) + self.D * u

    def forward(self, x):
        u, gate = self.inp(self.norm(x)).chunk(2, -1)
        u = self.conv(u.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)
        return x + self.drop(self.out(self.ssm(F.silu(u)) * F.silu(gate)))


def _encoder_layers(d, n_layers, heads, dropout):
    return nn.ModuleList(nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, 'gelu', batch_first=True, norm_first=True)
                         for _ in range(n_layers))


class _Temporal(nn.Module):
    """Shared step encoder for firm histories: x, m are (batch, K, features), most recent year last."""

    def __init__(self, n_features, d, dropout):
        super().__init__()
        self.step = nn.Sequential(nn.Linear(2 * n_features, 2 * d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.absent = nn.Parameter(torch.randn(d) * 0.02)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def embed(self, x, m, p):
        z = self.step(torch.cat([x, m], -1))
        return torch.where(p[..., None] > 0, z, self.absent)


class TemporalTransformer(_Temporal):
    def __init__(self, n_features, d=64, n_layers=2, heads=4, dropout=0.1, seq_len=5):
        super().__init__(n_features, d, dropout)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d) * 0.02)
        self.layers = _encoder_layers(d, n_layers, heads, dropout)

    def forward(self, x, m, p):
        z = self.embed(x, m, p) + self.pos
        for lyr in self.layers:
            z = lyr(z, src_key_padding_mask=p <= 0)
        return self.head(z[:, -1]).squeeze(-1)


class TemporalMamba(_Temporal):
    def __init__(self, n_features, d=64, n_layers=2, d_state=16, dropout=0.1, seq_len=5):
        super().__init__(n_features, d, dropout)
        self.layers = nn.ModuleList(MambaBlock(d, d_state, d_conv=2, dropout=dropout) for _ in range(n_layers))

    def forward(self, x, m, p):
        z = self.embed(x, m, p)
        for lyr in self.layers:
            z = lyr(z)
        return self.head(z[:, -1]).squeeze(-1)


ARCH = {'temp_trans': TemporalTransformer, 'temp_mamba': TemporalMamba}


def history_index(gvkey, years, k=5):
    """(rows, k) index of each row's own firm history, most recent last; -1 where a year is absent.
    Rows must be sorted by firm then year. A gap in fiscal years ends the history. Only the row
    itself and earlier years are referenced, so no future information enters a sequence."""
    n = len(gvkey)
    idx = np.full((n, k), -1, dtype=np.int64)
    idx[:, -1] = np.arange(n)
    for j in range(k - 2, -1, -1):
        cur = idx[:, j + 1]
        ok = cur > 0
        prev = cur - 1
        ok[ok] = (gvkey[prev[ok]] == gvkey[cur[ok]]) & (years[prev[ok]] == years[cur[ok]] - 1)
        idx[ok, j] = prev[ok]
    return idx


def stack_history(v, m, idx):
    """Turn per-row arrays into (rows, k, features) values and masks plus a (rows, k) presence mask."""
    p = (idx >= 0).astype(np.float32)
    vs, ms = v[np.maximum(idx, 0)], m[np.maximum(idx, 0)]
    return vs * p[..., None], ms * p[..., None], p


class DeepClassifier:
    """Deep ensemble of `n_members` networks. Epochs are a fixed hyper-parameter chosen on pre-2003
    folds, so the most recent training years are used for fitting rather than held out for early
    stopping (the zero-lag advice of Cai & Ye, ICML 2025)."""

    def __init__(self, arch, arch_kw=None, n_members=3, ratio=5, epochs=20, batch=256, lr=1e-3,
                 weight_decay=1e-5, seed=0):
        self.arch, self.arch_kw, self.n_members, self.ratio = arch, arch_kw or {}, n_members, ratio
        self.epochs, self.batch, self.lr, self.weight_decay, self.seed = epochs, batch, lr, weight_decay, seed

    def _train(self, net, inputs, y, rng, snapshot=None):
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        n_neg = min(len(neg), int(self.ratio * len(pos)))
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        target = torch.as_tensor(y, dtype=torch.float32)
        for ep in range(1, self.epochs + 1):
            net.train()
            order = rng.permutation(np.concatenate([pos, rng.choice(neg, n_neg, replace=False)]))
            for s in range(0, len(order), self.batch):
                b = order[s:s + self.batch]
                if len(b) < 2:
                    continue
                out = net(*[torch.from_numpy(a[b]) for a in inputs])
                t = target[b]
                loss = F.binary_cross_entropy_with_logits(out, t if out.ndim == 1 else t[:, None].expand_as(out))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            if snapshot is not None:
                snapshot(ep, net)

    @torch.no_grad()
    def _score(self, net, inputs, batch=4096):
        net.eval()
        out = []
        for s in range(0, len(inputs[0]), batch):
            o = torch.sigmoid(net(*[torch.from_numpy(a[s:s + batch]) for a in inputs]))
            out.append((o if o.ndim == 1 else o.mean(1)).numpy())
        return np.concatenate(out)

    def fit(self, inputs_tr, y, snapshot=None):
        y = np.asarray(y)
        n_features = inputs_tr[0].shape[-1]
        self.nets_ = []
        for b in range(self.n_members):
            torch.manual_seed(self.seed + b)
            rng = np.random.RandomState(self.seed + b)
            net = ARCH[self.arch](n_features, **self.arch_kw)
            self._train(net, inputs_tr, y, rng, snapshot)
            self.nets_.append(net)
        return self

    def predict_proba(self, inputs):
        return np.mean([self._score(n, inputs) for n in self.nets_], axis=0)

    def fit_predict(self, inputs_tr, y, inputs_te, eval_epochs=None):
        """Fit and score test rows. With eval_epochs, returns {epoch: scores} for epoch selection."""
        kept = {}

        def snap(ep, n):
            if eval_epochs and ep in eval_epochs:
                kept.setdefault(ep, []).append(self._score(n, inputs_te))

        self.fit(inputs_tr, y, snap)
        return {ep: np.mean(v, axis=0) for ep, v in kept.items()} if eval_epochs else self.predict_proba(inputs_te)


# Frozen on the pre-2003 folds (results/experiments/cv_deep.json); see experiments/PROTOCOL.md.
TEMPORAL_MEMBER = dict(arch='temp_trans', arch_kw=dict(d=32, n_layers=2, dropout=0.2), n_members=3, epochs=15,
                       ratio=10, lr=1e-3, weight_decay=1e-5)
HISTORY_YEARS = 5


class TemporalMember:
    """Ensemble member that reads each firm's last HISTORY_YEARS fiscal years. Unlike the other
    members it needs firm and year keys, so it takes the winsorised frame rather than a row matrix."""

    def __init__(self, features, seed=0):
        self.features, self.seed = list(features), seed

    def _history(self, D, gvkey, years, rows):
        v, m = self.prep_.transform(D[self.features].values)
        return stack_history(v, m, history_index(gvkey, years, HISTORY_YEARS)[rows])

    def fit(self, D, gvkey, years, tr, y_tr):
        self.prep_ = Preprocessor().fit(D[self.features].values[tr])
        self.clf_ = DeepClassifier(seed=self.seed, **TEMPORAL_MEMBER).fit(self._history(D, gvkey, years, tr), y_tr)
        return self

    def predict_scores(self, D, gvkey, years, rows):
        return self.clf_.predict_proba(self._history(D, gvkey, years, rows))

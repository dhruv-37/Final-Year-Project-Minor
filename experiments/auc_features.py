"""Extra inputs and model variants for PROTOCOL_AUC.md. Features use a firm's earlier years or the same
fiscal year's cross-section only, so no future information enters a row."""
import math

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from torch import nn
from torch.nn import functional as F

from deep_models import ARCH, DeepClassifier, TemporalTransformer
from models import BalancedBagger

PEER_INPUTS = ['act_ta_yr', 'ppe_ta_yr', 'invt_ta_yr', 'rect_ta_yr', 'margin_yr', 'asset_turn_yr', 'log_at_yr', 'dp_ta_yr',
               'lev_yr']
KEY_RANKS = [b + '_yr' for b in ('soft_assets', 'dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'ch_cs', 'ch_roa', 'bm', 'reoa',
                                 'EBIT', 'roa', 'margin', 'asset_turn', 'lev', 'cash_ta', 'rect_sale', 'inv_cogs',
                                 'mve_ta', 'fin_need', 'tata')]


def spell_ids(gvkey, years):
    """One id per unbroken run of fiscal years of a firm. Rows are sorted by firm, then year."""
    return np.cumsum(np.r_[True, (gvkey[1:] != gvkey[:-1]) | (years[1:] != years[:-1] + 1)])


def _year_rank(s, years):
    return s.replace([np.inf, -np.inf], np.nan).groupby(years).rank(pct=True)


def _clip_by_year(s, years):
    g = s.groupby(years)
    return s.clip(g.transform(lambda v: v.quantile(0.01)), g.transform(lambda v: v.quantile(0.99)))


def market_features(D, gvkey, years):
    spell = spell_ids(gvkey, years)
    d = D.reset_index(drop=True)
    mve = (d['prcc_f'] * d['csho']).where(lambda s: s > 0)
    g = mve.groupby(spell)
    mve_g = mve / g.shift(1) - 1
    asset_g = d['at'] / d['at'].groupby(spell).shift(1) - 1
    sale_g = d['sale'] / d['sale'].groupby(spell).shift(1) - 1
    raw = {'mkt_g': mve_g, 'mkt_g_lag': mve_g.groupby(spell).shift(1), 'mkt_g3': mve / g.shift(3) - 1,
           'mkt_vol': mve_g.groupby(spell).rolling(4, min_periods=2).std().reset_index(level=0, drop=True).sort_index(),
           'shares_g': d['csho'] / d['csho'].groupby(spell).shift(1) - 1, 'mkt_vs_assets': mve_g - asset_g, 'mkt_vs_sales': mve_g - sale_g}
    return pd.DataFrame({k + '_yr': _year_rank(v, years) for k, v in raw.items()}).astype(np.float32)


def peer_labels(D, years, k=20, seed=0):
    """Per-year clusters on business structure, standing in for the industry codes this dataset lacks."""
    labels = np.zeros(len(D), dtype=np.int32)
    X = D[PEER_INPUTS].fillna(0.5).values
    for yr in np.unique(years):
        rows = np.where(years == yr)[0]
        labels[rows] = KMeans(n_clusters=min(k, len(rows)), n_init=3, random_state=seed).fit_predict(X[rows])
    return labels


def peer_features(D, years, labels):
    d = D[KEY_RANKS].reset_index(drop=True)
    return (d - d.groupby([years, labels]).transform('median')).add_suffix('_peer').astype(np.float32)


def accrual_features(D, gvkey, years, labels, min_rows=15):
    """Modified-Jones style: accruals not explained by 1/assets, cash sales growth and PPE, among peers."""
    spell = spell_ids(gvkey, years)
    d = D.reset_index(drop=True)
    at_l = d['at'].groupby(spell).shift(1).where(lambda s: s > 0)
    dsale = d['sale'] - d['sale'].groupby(spell).shift(1)
    drect = d['rect'] - d['rect'].groupby(spell).shift(1)
    R = pd.DataFrame({'inv': 1 / at_l, 'sales': (dsale - drect) / at_l, 'ppe': d['ppegt'] / at_l}).replace([np.inf, -np.inf], np.nan)
    R = pd.DataFrame({col: _clip_by_year(R[col], years) for col in R})
    out = {}
    for tgt in ('tata', 'ch_rsst'):
        y = _clip_by_year(d[tgt], years)
        res = np.full(len(d), np.nan)
        ok = (R.notna().all(axis=1) & y.notna()).values
        for yr in np.unique(years):
            in_year = np.where((years == yr) & ok)[0]
            if len(in_year) < min_rows:
                continue
            for lab in np.unique(labels[in_year]):
                rows = in_year[labels[in_year] == lab]
                if len(rows) < min_rows:
                    rows = in_year
                A = np.c_[np.ones(len(rows)), R.values[rows]]
                beta = np.linalg.lstsq(A, y.values[rows], rcond=None)[0]
                keep = rows[labels[rows] == lab]
                res[keep] = y.values[keep] - np.c_[np.ones(len(keep)), R.values[keep]] @ beta
        res = pd.Series(res, index=d.index)
        out[f'acc_{tgt}_yr'] = _year_rank(res, years)
        out[f'acc_{tgt}_abs_yr'] = _year_rank(res.abs(), years)
    return pd.DataFrame(out).astype(np.float32)


def surprise_features(D, gvkey, years, tr, y_tr, inputs):
    """Actual rank minus what the firm's prior year predicts; fit on the window's non-fraud rows."""
    spell = spell_ids(gvkey, years)
    prev = np.r_[-1, np.arange(len(years) - 1)]
    prev[np.r_[True, spell[1:] != spell[:-1]]] = -1
    X = np.nan_to_num(D[inputs].values[np.maximum(prev, 0)], nan=0.5)
    T = D[KEY_RANKS].values
    fit = tr[(y_tr == 0) & (prev[tr] >= 0)]
    pred = Ridge(alpha=10.0).fit(X[fit], np.nan_to_num(T[fit], nan=0.5)).predict(X)
    res = T - pred
    res[prev < 0] = np.nan
    return pd.DataFrame(res, columns=[c + '_surp' for c in KEY_RANKS], index=D.index).astype(np.float32)


class WeightedBagger(BalancedBagger):
    """BalancedBagger whose models also take per-row weights."""

    def fit(self, X, y, w):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
        rng = np.random.RandomState(self.seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        n_neg = min(len(neg), int(self.ratio * len(pos)))
        self.models_ = []
        for b in range(self.n_bags):
            idx = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
            self.models_.append(self.make_model(self.seed + b).fit(X[idx], y[idx], sample_weight=w[idx]))
        self.make_model = None
        return self


class ModulatedTT(TemporalTransformer):
    """Input-level time modulation: each feature gets a scale and bias from a year embedding."""

    def __init__(self, n_features, d=64, n_layers=2, heads=4, dropout=0.1, seq_len=5):
        super().__init__(n_features, d, n_layers, heads, dropout, seq_len)
        self.mod = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2 * n_features))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x, m, p, t):
        psi = torch.stack([t, t * t, torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)], -1)
        gamma, beta = self.mod(psi).chunk(2, -1)
        return super().forward(x * (1 + gamma) + beta, m, p)


ARCH['mod_tt'] = ModulatedTT


class PairwiseClassifier(DeepClassifier):
    """Optimises a surrogate of AUC directly: logistic loss on every fraud / non-fraud pair in a batch."""

    def _train(self, net, inputs, y, rng, snapshot=None):
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        n_neg = min(len(neg), int(self.ratio * len(pos)))
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        target = torch.as_tensor(y)
        for ep in range(1, self.epochs + 1):
            net.train()
            order = rng.permutation(np.concatenate([pos, rng.choice(neg, n_neg, replace=False)]))
            for s in range(0, len(order), self.batch):
                b = order[s:s + self.batch]
                t = target[b]
                if t.sum() == 0 or t.sum() == len(b):
                    continue
                out = net(*[torch.from_numpy(a[b]) for a in inputs])
                loss = F.softplus(out[t == 0][None, :] - out[t == 1][:, None]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            if snapshot is not None:
                snapshot(ep, net)

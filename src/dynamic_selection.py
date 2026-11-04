"""Feature selectors for experiments/PROTOCOL_FS.md. Each runs on one training window's rows only and
scores features on out-of-time inner folds that copy the outer protocol (feature_selection.inner_folds).

    stability_frequency   Meinshausen & Buhlmann (2010): how often a feature ranks high on firm subsamples
    shadow_frequency      Boruta (Kursa & Rudnicki 2010): how often a feature beats permuted copies of itself
    permutation_drop      after Altmann et al. (2010): AUC lost when a feature is shuffled out of time
    adversarial_drift     Uber AVFS (arXiv 2004.03045): how much a feature gives away the recent years
    mrmr_order            Ding & Peng (2005): relevance over redundancy, greedily
    SoftBagger            soft selection: every bag samples features by importance, none is removed for good
    history_features      changes against a firm's own past, to widen the candidate pool
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from feature_selection import _shap_importance, _share, inner_folds
from models import BalancedBagger, lgbm_factory, mask_serial


def _bagger(p, X, y, seed, n_jobs, n_bags=2):
    return BalancedBagger(lgbm_factory(p, n_jobs), n_bags, p['ratio'], seed).fit(X, y)


def oot_importance(X, y, years, paaer, p, seed, n_jobs):
    """Mean |SHAP| share on out-of-time rows."""
    X = np.asarray(X, dtype=np.float32)
    imp = np.zeros(X.shape[1])
    for tr, va in inner_folds(years):
        gb = _bagger(p, X[tr], mask_serial(y, paaer, tr, va), seed, n_jobs)
        imp += _share(_shap_importance(gb, X[va]))
    return imp / 3


def stability_frequency(X, y, years, paaer, gvkey, p, seed, n_jobs, n_runs=30, top=0.5):
    """Share of runs in which a feature is in the top half by out-of-time |SHAP|, on half of the firms."""
    X = np.asarray(X, dtype=np.float32)
    rng = np.random.RandomState(seed)
    folds = inner_folds(years)
    hits = np.zeros(X.shape[1])
    k = int(top * X.shape[1])
    for b in range(n_runs):
        tr, va = folds[b % len(folds)]
        firms = np.unique(gvkey[tr])
        sub = tr[np.isin(gvkey[tr], rng.choice(firms, len(firms) // 2, replace=False))]
        gb = _bagger(p, X[sub], mask_serial(y, paaer, sub, va), seed + b, n_jobs, n_bags=1)
        hits[np.argsort(_shap_importance(gb, X[va]))[-k:]] += 1
    return hits / n_runs


def _shuffle_within_year(X, years, gen):
    out = X.copy()
    for yr in np.unique(years):
        rows = np.where(years == yr)[0]
        out[rows] = gen.permuted(X[rows], axis=0)
    return out


def shadow_frequency(X, y, years, paaer, p, seed, n_jobs, quantiles=(1.0, 0.95, 0.75), n_runs=12):
    """{quantile: share of runs a feature's out-of-time |SHAP| beats that quantile of the shadows}."""
    X = np.asarray(X, dtype=np.float32)
    gen = np.random.default_rng(seed)
    folds = inner_folds(years)
    n = X.shape[1]
    hits = {q: np.zeros(n) for q in quantiles}
    for b in range(n_runs):
        tr, va = folds[b % len(folds)]
        both = np.hstack([X, _shuffle_within_year(X, years, gen)])
        gb = _bagger(p, both[tr], mask_serial(y, paaer, tr, va), seed + b, n_jobs, n_bags=1)
        imp = _shap_importance(gb, both[va])
        for q in quantiles:
            hits[q] += imp[:n] > np.quantile(imp[n:], q)
    return {q: v / n_runs for q, v in hits.items()}


def permutation_drop(X, y, years, paaer, p, seed, n_jobs, repeats=3):
    """Mean out-of-time AUC drop per feature and its standard error across the inner folds."""
    X = np.asarray(X, dtype=np.float32)
    rng = np.random.RandomState(seed)
    drops = []
    for tr, va in inner_folds(years):
        gb = _bagger(p, X[tr], mask_serial(y, paaer, tr, va), seed, n_jobs, n_bags=3)
        Xv, yv = X[va].copy(), y[va]
        base = roc_auc_score(yv, gb.predict_proba(Xv))
        d = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            keep = Xv[:, j].copy()
            for _ in range(repeats):
                Xv[:, j] = rng.permutation(keep)
                d[j] += base - roc_auc_score(yv, gb.predict_proba(Xv))
            Xv[:, j] = keep
        drops.append(d / repeats)
    drops = np.array(drops)
    return drops.mean(0), drops.std(0, ddof=1) / np.sqrt(len(drops))


def adversarial_drift(X, years, seed, n_jobs, recent=3, max_rows=30000):
    """|SHAP| share of a model that tells the window's last `recent` years from the earlier ones."""
    X = np.asarray(X, dtype=np.float32)
    rng = np.random.RandomState(seed)
    rows = rng.choice(len(X), min(max_rows, len(X)), replace=False)
    label = (years[rows] > years.max() - recent).astype(int)
    m = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, colsample_bytree=0.5, subsample=0.8,
                       subsample_freq=1, random_state=seed, n_jobs=n_jobs, verbose=-1).fit(X[rows], label)
    return _share(np.abs(m.booster_.predict(X[rows], pred_contrib=True)[:, :-1]).mean(0))


def mrmr_order(X, imp, seed, max_rows=20000):
    """All feature indices, best first: importance over mean |Spearman| with those already chosen."""
    rng = np.random.RandomState(seed)
    rows = rng.choice(len(X), min(max_rows, len(X)), replace=False)
    corr = pd.DataFrame(X[rows]).corr(method='spearman').abs().fillna(0).values
    order = [int(np.argmax(imp))]
    left = set(range(len(imp))) - set(order)
    while left:
        cand = np.array(sorted(left))
        score = imp[cand] / np.maximum(corr[np.ix_(cand, order)].mean(1), 1e-3)
        pick = int(cand[np.argmax(score)])
        order.append(pick)
        left.remove(pick)
    return order


class SoftBagger:
    """BalancedBagger whose bags each keep `frac` of the features, drawn with probability imp ** alpha."""

    def __init__(self, make_model, weights, alpha=1.0, frac=0.7, n_bags=10, ratio=10, seed=0):
        self.make_model, self.w, self.alpha, self.frac = make_model, np.asarray(weights), alpha, frac
        self.n_bags, self.ratio, self.seed = n_bags, ratio, seed

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
        rng = np.random.RandomState(self.seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        n_neg = min(len(neg), int(self.ratio * len(pos)))
        prob = np.maximum(self.w, 1e-6) ** self.alpha
        prob = prob / prob.sum()
        self.models_ = []
        for b in range(self.n_bags):
            idx = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
            cols = np.sort(rng.choice(X.shape[1], int(self.frac * X.shape[1]), replace=False, p=prob))
            self.models_.append((cols, self.make_model(self.seed + b).fit(X[np.ix_(idx, cols)], y[idx])))
        self.make_model = None
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.mean([m.predict_proba(X[:, cols])[:, 1] for cols, m in self.models_], axis=0)


def history_features(frame, gvkey, years, bases, window=4):
    """For each base column: deviation from the firm's own mean over its previous `window` years, that
    deviation in units of the firm's own volatility, and the volatility. Rows must be sorted by firm then
    year. Only earlier years enter, and a gap in fiscal years starts a new history."""
    new_spell = np.r_[True, (gvkey[1:] != gvkey[:-1]) | (years[1:] != years[:-1] + 1)]
    spell = np.cumsum(new_spell)
    now = frame[bases].reset_index(drop=True)
    past = now.groupby(spell).shift(1)
    roll = past.groupby(spell).rolling(window, min_periods=2)
    mean = roll.mean().reset_index(level=0, drop=True).sort_index()
    vol = roll.std().reset_index(level=0, drop=True).sort_index()
    dev = now - mean
    z = (dev / vol.where(vol > 1e-3)).clip(-5, 5)
    return pd.concat([dev.add_suffix('_dev'), z.add_suffix('_z'), vol.add_suffix('_vol')], axis=1).astype(np.float32)

"""Dynamic recursive feature elimination (DRFE), run separately inside every training window.

"Dynamic" in two senses:
  * the step size adapts: each round drops the weakest features whose combined importance stays
    under `tau`, so useless features go in big jumps and the steps shrink as only signal is left;
  * the subset is re-selected per expanding window, using that window's own rows only.

Importance is recomputed every round on out-of-time rows and is a consensus of the tree and linear
members (mean |SHAP| and |standardised coefficient|), each weighted by its validation AUC. Every
subset is scored on inner folds that copy the outer protocol (2-year gap, serial-fraud masking).
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from evaluate import rank_norm
from models import BalancedBagger, LogReg, lgbm_factory, mask_serial


def inner_folds(years, n_val=3, gap=2):
    """Validate on each of the window's last n_val years, training up to `gap` years before it."""
    last = int(years.max())
    return [(np.where(years <= v - gap)[0], np.where(years == v)[0]) for v in range(last, last - n_val, -1)]


def _shap_importance(bagger, X):
    return np.mean([np.abs(m.booster_.predict(X, pred_contrib=True)[:, :-1]).mean(0) for m in bagger.models_], axis=0)


def _lr_importance(lr):
    """|coefficient| per input column. The imputer drops columns with no value in the training rows (a
    3-year market feature in the earliest years); those get 0, since the model cannot use them."""
    kept = ~np.isnan(lr.pipe_[0].statistics_)
    out = np.zeros(len(kept))
    out[kept] = np.abs(lr.pipe_[-1].coef_[0])
    return out


def _share(v):
    return v / max(v.sum(), 1e-12)


def plan_drop(imp, tau, max_frac, floor=1):
    """Number of weakest features to drop: as many as fit in the importance budget tau."""
    s = np.sort(_share(imp))
    n = int(np.searchsorted(np.cumsum(s), tau, side='right'))
    return int(np.clip(n, floor, max(floor, int(max_frac * len(imp)))))


def choose_subset(path, rule='one_se'):
    """rule='best': highest mean score; rule='one_se': fewest features within one SE of the best."""
    best = max(path, key=lambda r: r['auc'])
    if rule == 'best':
        return best['features']
    ok = [r for r in path if r['auc'] >= best['auc'] - best['se']]
    return min(ok, key=lambda r: len(r['features']))['features']


class DynamicRFE:
    """fit(X, y, years, paaer, names) records the elimination `path_`; use choose_subset to pick."""

    def __init__(self, lgbm_params, tau=0.05, max_frac=0.25, min_features=15, n_bags=2, patience=4,
                 seed=0, n_jobs=-1):
        self.p, self.tau, self.max_frac, self.min_features = lgbm_params, tau, max_frac, min_features
        self.n_bags, self.patience, self.seed, self.n_jobs = n_bags, patience, seed, n_jobs

    def _round(self, X, y, folds, paaer, cols):
        """Score one subset on the inner folds; returns fold AUCs and the consensus importance."""
        imp, aucs = np.zeros(len(cols)), []
        for tr, va in folds:
            ytr = mask_serial(y, paaer, tr, va)
            Xtr, Xva = X[np.ix_(tr, cols)], X[np.ix_(va, cols)]
            gb = BalancedBagger(lgbm_factory(self.p, self.n_jobs), self.n_bags, self.p['ratio'], self.seed).fit(Xtr, ytr)
            lr = LogReg().fit(Xtr, ytr)
            sg, sl = gb.predict_proba(Xva), lr.predict_proba(Xva)
            ag, al = roc_auc_score(y[va], sg), roc_auc_score(y[va], sl)
            aucs.append(roc_auc_score(y[va], rank_norm(sg) + rank_norm(sl)))
            w = np.array([max(ag - 0.5, 1e-3), max(al - 0.5, 1e-3)])
            imp += (w[0] * _share(_shap_importance(gb, Xva.astype(np.float32))) + w[1] * _share(_lr_importance(lr))) / w.sum()
        return np.array(aucs), imp / len(folds)

    def fit(self, X, y, years, paaer, names):
        X = np.asarray(X, dtype=np.float32)
        folds = inner_folds(years)
        cols, self.path_ = np.arange(X.shape[1]), []
        best, stale = -1.0, 0
        while True:
            aucs, imp = self._round(X, y, folds, paaer, cols)
            self.path_.append({'features': [names[j] for j in cols], 'auc': float(aucs.mean()),
                               'se': float(aucs.std(ddof=1) / np.sqrt(len(aucs)))})
            stale = 0 if aucs.mean() > best else stale + 1
            best = max(best, float(aucs.mean()))
            if len(cols) <= self.min_features or stale >= self.patience:
                return self
            n_drop = min(plan_drop(imp, self.tau, self.max_frac), len(cols) - self.min_features)
            cols = cols[np.sort(np.argsort(imp)[n_drop:])]

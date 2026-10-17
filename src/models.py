"""Models: faithful RUSBoost baseline (Bao et al.), balanced-bagged LightGBM / XGBoost,
regularised logistic regression, and a rank-average ensemble.

Why balanced bagging?  Only ~0.7% of firm-years are fraud. Each bag keeps ALL frauds and a
random subset of non-frauds (ratio tuned), so every bag trains on a few thousand rows
(fast, CPU-only) while the bags together still see most non-fraud rows.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from evaluate import rank_norm


class BalancedBagger:
    def __init__(self, make_model, n_bags=10, ratio=10, seed=0):
        self.make_model, self.n_bags, self.ratio, self.seed = make_model, n_bags, ratio, seed

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
        rng = np.random.RandomState(self.seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        n_neg = min(len(neg), int(self.ratio * len(pos)))
        self.models_ = []
        for b in range(self.n_bags):
            idx = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
            m = self.make_model(self.seed + b)
            m.fit(X[idx], y[idx])
            self.models_.append(m)
        self.make_model = None          # closures cannot be pickled; not needed after fit
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.mean([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)


# ------------------------------------------------------------------ factories
def lgbm_factory(p, n_jobs=-1):
    def make(seed):
        return LGBMClassifier(
            n_estimators=p['n_estimators'], learning_rate=p['learning_rate'],
            num_leaves=p['num_leaves'], min_child_samples=p['min_child_samples'],
            colsample_bytree=p['colsample_bytree'], subsample=p['subsample'], subsample_freq=1,
            reg_lambda=p['reg_lambda'], random_state=seed, n_jobs=n_jobs, verbose=-1)
    return make


def xgb_factory(p, n_jobs=-1):
    def make(seed):
        return XGBClassifier(
            n_estimators=p['n_estimators'], learning_rate=p['learning_rate'],
            max_depth=p['max_depth'], min_child_weight=p['min_child_weight'],
            colsample_bytree=p['colsample_bytree'], subsample=p['subsample'],
            reg_lambda=p['reg_lambda'], tree_method='hist', random_state=seed,
            n_jobs=n_jobs, verbosity=0)
    return make


def sample_lgbm(rng):
    return dict(n_estimators=int(rng.randint(100, 500)),
                learning_rate=float(10 ** rng.uniform(-2, -1)),
                num_leaves=int(rng.randint(4, 32)), min_child_samples=int(rng.randint(10, 100)),
                colsample_bytree=float(rng.uniform(0.4, 1)), subsample=float(rng.uniform(0.5, 1)),
                reg_lambda=float(10 ** rng.uniform(-1, 1.5)),
                ratio=int(rng.choice([3, 5, 10, 20])))


def sample_xgb(rng):
    return dict(n_estimators=int(rng.randint(100, 400)),
                learning_rate=float(10 ** rng.uniform(-2, -0.9)),
                max_depth=int(rng.randint(2, 7)), min_child_weight=float(rng.uniform(1, 10)),
                colsample_bytree=float(rng.uniform(0.4, 1)), subsample=float(rng.uniform(0.5, 1)),
                reg_lambda=float(10 ** rng.uniform(-1, 1.5)),
                ratio=int(rng.choice([3, 5, 10, 20])))


class RUSBoost:
    """RUSBoost (Seiffert et al. 2010) = AdaBoost.M2 + random under-sampling each round.

    Re-implemented because the scikit-learn/imbalanced-learn version collapsed to a single
    effective tree on this data. Defaults mirror the authors' MATLAB code of Bao et al.:
    300 trees, min leaf size 5, learning rate 0.1, 1:1 under-sampling, and MATLAB's boosted-tree
    template default MaxNumSplits=10 (-> max_leaf_nodes=11, grown best-first).
    """

    def __init__(self, n_estimators=300, min_samples_leaf=5, learning_rate=0.1, seed=0):
        self.n_estimators, self.min_samples_leaf = n_estimators, min_samples_leaf
        self.learning_rate, self.seed = learning_rate, seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        self.imp_ = SimpleImputer(strategy='median').fit(X)
        X = self.imp_.transform(X)
        n = len(y)
        rng = np.random.RandomState(self.seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        D = np.full(n, 1.0 / n)
        self.trees_, self.alphas_ = [], []
        for _ in range(self.n_estimators):
            sel = np.concatenate([pos, rng.choice(neg, len(pos), replace=False)])
            w = D[sel] / D[sel].sum()
            tree = DecisionTreeClassifier(min_samples_leaf=self.min_samples_leaf, max_leaf_nodes=11,
                                          random_state=int(rng.randint(1 << 30)))
            tree.fit(X[sel], y[sel], sample_weight=w)
            h_true = tree.predict_proba(X)[np.arange(n), y]
            eps = float((D * (1.0 - h_true)).sum())          # 2-class pseudo-loss
            if eps >= 0.5:
                continue                                      # worse than chance: skip learner
            eps = max(eps, 1e-10)
            beta = eps / (1.0 - eps)
            D = D * beta ** (self.learning_rate * h_true)
            D /= D.sum()
            self.trees_.append(tree)
            self.alphas_.append(self.learning_rate * np.log(1.0 / beta))
        return self

    def predict_proba(self, X):
        X = self.imp_.transform(np.asarray(X, dtype=np.float64))
        s = np.zeros(len(X))
        for t, a in zip(self.trees_, self.alphas_):
            s += a * t.predict_proba(X)[:, 1]
        return s / max(sum(self.alphas_), 1e-12)


class LogReg:
    """Regularised logistic regression on (winsorised) features - a simple, interpretable member."""

    def __init__(self, C=0.05):
        self.C = C

    def fit(self, X, y):
        self.pipe_ = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                   LogisticRegression(C=self.C, class_weight='balanced',
                                                      max_iter=2000))
        self.pipe_.fit(np.asarray(X, dtype=np.float64), np.asarray(y))
        return self

    def predict_proba(self, X):
        return self.pipe_.predict_proba(np.asarray(X, dtype=np.float64))[:, 1]


# ------------------------------------------------------------------ serial-fraud handling
def mask_serial(y, paaer, tr, ev):
    """Authors' rule: training rows that belong to an AAER case also present among the
    evaluation-period frauds get label 0 (the case is not yet 'known' at training time)."""
    ytr = y[tr].copy()
    cases = pd.Series(paaer[ev][y[ev] == 1]).dropna().unique()
    ytr[np.isin(paaer[tr], cases)] = 0
    return ytr


# ------------------------------------------------------------------ rolling-origin CV
def make_folds(years, last_train_years=(1997, 1998, 1999), gap=1):
    """Train on all years <= T, skip `gap` year(s), validate on T+gap+1 (no look-ahead)."""
    folds = []
    for T in last_train_years:
        tr = np.where((years >= 1991) & (years <= T))[0]
        va = np.where(years == T + gap + 1)[0]
        folds.append((tr, va))
    return folds


def cv_predict(fit_predict, Xs, y, folds, paaer=None):
    """fit_predict(Xtr, ytr, Xva) -> scores; Xs[k] is the full matrix as winsorised for fold k.
    Returns [(va_idx, scores)] and mean AUC."""
    preds, aucs = [], []
    for (tr, va), X in zip(folds, Xs):
        ytr = mask_serial(y, paaer, tr, va) if paaer is not None else y[tr]
        s = fit_predict(X[tr], ytr, X[va])
        preds.append((va, s))
        aucs.append(roc_auc_score(y[va], s))
    return preds, float(np.mean(aucs))


def random_search(sampler, build, Xs, y, folds, n_trials, seed=0, n_bags=4, paaer=None, log=print):
    """Random search on the rolling-origin folds; `build(params, n_bags)` returns a model."""
    rng = np.random.RandomState(seed)
    best, best_auc, history = None, -1, []
    for t in range(n_trials):
        p = sampler(rng)

        def fp(Xtr, ytr, Xva, p=p):
            return build(p, n_bags).fit(Xtr, ytr).predict_proba(Xva)

        _, auc = cv_predict(fp, Xs, y, folds, paaer)
        history.append({**p, 'cv_auc': auc})
        if auc > best_auc:
            best, best_auc = p, auc
        log(f'  trial {t + 1:>2}/{n_trials}  cv AUC={auc:.4f}  best={best_auc:.4f}')
    return best, best_auc, pd.DataFrame(history)


def ensemble_scores(score_list, weights=None):
    r = np.array([rank_norm(s) for s in score_list])
    w = np.ones(len(r)) / len(r) if weights is None else np.asarray(weights) / np.sum(weights)
    return (r * w[:, None]).sum(0)

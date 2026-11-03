"""Focal positive-unlabeled boosting: the iFPU risk of Zavitsanos & Paliouras (arXiv 2605.14467).

AAER labels are positive-unlabeled: a label of 0 means "not caught (yet)", not "clean". With class
prior pi, positives P and unlabeled U (Eq. 11 of the paper):

    risk = pi * mean_P[-(1-g)^gamma log g] + max(0, mean_U[-g^gamma log(1-g)] - pi * mean_P[-g^gamma log(1-g)])
"""
import numpy as np
from xgboost import XGBClassifier


def _pos_grad(f, gamma):
    """d/df of -(1-g)^gamma log g."""
    g = np.clip(1 / (1 + np.exp(-f)), 1e-7, 1 - 1e-7)
    return gamma * g * (1 - g) ** gamma * np.log(g) - (1 - g) ** (gamma + 1)


def _neg_grad(f, gamma):
    """d/df of -g^gamma log(1-g)."""
    g = np.clip(1 / (1 + np.exp(-f)), 1e-7, 1 - 1e-7)
    return g ** (gamma + 1) - gamma * g ** gamma * (1 - g) * np.log(1 - g)


def _neg_loss(f, gamma):
    g = np.clip(1 / (1 + np.exp(-f)), 1e-7, 1 - 1e-7)
    return -(g ** gamma) * np.log(1 - g)


class FocalPUObjective:
    """XGBoost objective: gradient of the iFPU risk, scaled by n to match XGBoost's sum convention.
    The hessian is the absolute finite difference of the gradient, floored for stability."""

    def __init__(self, pi=0.02, gamma=3.0, eps=1e-3):
        self.pi, self.gamma, self.eps = pi, gamma, eps

    def grad(self, y, f):
        pos, n = y == 1, len(y)
        w_p, w_u = n * self.pi / max(pos.sum(), 1), n / max((~pos).sum(), 1)
        u_risk = _neg_loss(f[~pos], self.gamma).mean() - self.pi * _neg_loss(f[pos], self.gamma).mean()
        on = float(u_risk > 0)                        # the non-negative clip of nnPU
        out = np.empty(n)
        out[pos] = w_p * (_pos_grad(f[pos], self.gamma) - on * _neg_grad(f[pos], self.gamma))
        out[~pos] = on * w_u * _neg_grad(f[~pos], self.gamma)
        return out

    def __call__(self, y, f):
        g = self.grad(y, f)
        h = np.abs(self.grad(y, f + self.eps) - g) / self.eps
        return g, np.maximum(h, 1e-6)


class PUXGB:
    """Mean of n_models XGBoost runs on the full training window under the focal PU risk."""

    def __init__(self, params, pi=0.02, gamma=3.0, n_models=3, seed=0, n_jobs=-1):
        self.p, self.pi, self.gamma, self.n_models, self.seed, self.n_jobs = params, pi, gamma, n_models, seed, n_jobs

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
        p = self.p
        self.models_ = [XGBClassifier(objective=FocalPUObjective(self.pi, self.gamma), base_score=self.pi,
                                      n_estimators=p['n_estimators'], learning_rate=p['learning_rate'],
                                      max_depth=p['max_depth'], min_child_weight=p['min_child_weight'],
                                      colsample_bytree=p['colsample_bytree'], subsample=p['subsample'],
                                      reg_lambda=p['reg_lambda'], tree_method='hist', random_state=self.seed + b,
                                      n_jobs=self.n_jobs, verbosity=0).fit(X, y)
                        for b in range(self.n_models)]
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.mean([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)

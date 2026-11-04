"""Why some papers report AUC above 0.8: the adopted trees and logistic regression, without the Temporal
Transformer, scored under weaker evaluation rules. PROTOCOL_AUC.md allows this as a diagnostic: these
numbers are never candidates and choose nothing.

    python experiments/diagnostic_splits.py

clean            expanding window, 2-year gap, serial-fraud masking (the project's rule)
no_mask          the same without serial-fraud masking
no_mask_no_gap   also trains up to the year before the test year
random_rows      random 5-fold split of firm-years 1991-2012: a firm's later years can train on its earlier
                 ones and the reverse, and fraud cases spanning years fall on both sides
random_firms     random 5-fold split by firm: the time order is lost, but no firm is on both sides
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from common import N_BAGS, OUT, TEST_YEARS, ctx, load, mask_serial, os, path, run_tasks, winsorized
from features import market_features
from models import TREE_LGBM, TREE_XGB, LogReg, WeightedBagger, ensemble_scores, lgbm_factory, recency_weights, xgb_factory
from run_reference import MEMBERS

SEED = 42
TEMPORAL = {'clean': (2, True), 'no_mask': (2, False), 'no_mask_no_gap': (1, False)}
RANDOM = ('random_rows', 'random_firms')
N_FOLDS = 5


def windows(rule):
    """[(name, train rows, evaluation rows)] of a rule."""
    c = ctx()
    if rule in TEMPORAL:
        return [(str(t), np.where(c.years <= t - TEMPORAL[rule][0])[0], np.where(c.years == t)[0]) for t in TEST_YEARS]
    rows = np.where(c.years <= max(TEST_YEARS))[0]
    rng = np.random.RandomState(SEED)
    units = c.gvkey[rows] if rule == 'random_firms' else rows
    uniq = rng.permutation(np.unique(units))
    fold = dict(zip(uniq, np.arange(len(uniq)) % N_FOLDS))
    assign = np.array([fold[u] for u in units])
    return [(str(k), rows[assign != k], rows[assign == k]) for k in range(N_FOLDS)]


def fit_task(rule, window, member, n_jobs):
    c = ctx()
    name, tr, te = next(w for w in windows(rule) if w[0] == window)
    ytr = mask_serial(c.y, c.paaer, tr, te) if rule in TEMPORAL and TEMPORAL[rule][1] else c.y[tr]
    X = np.c_[winsorized(tr)[c.feats].values, market_features(c.DW, c.gvkey, c.years).values]
    if member == 'LR':
        return LogReg().fit(X[tr], ytr).predict_proba(X[te])
    p = TREE_LGBM if member == 'LGBM' else TREE_XGB
    make, sd = (lgbm_factory(p, n_jobs), SEED) if member == 'LGBM' else (xgb_factory(p, n_jobs), SEED + 1)
    return WeightedBagger(make, N_BAGS, p['ratio'], sd).fit(X[tr], ytr, recency_weights(c.years[tr])).predict_proba(X[te])


def task_path(rule, window, member):
    return path('diagnostic', f'{rule}_{window}_{member}.pkl')


def main():
    c = ctx()
    rules = list(TEMPORAL) + list(RANDOM)
    run_tasks([(fit_task, task_path(r, w[0], m), dict(rule=r, window=w[0], member=m))
               for r in rules for w in windows(r) for m in MEMBERS], n_workers=5, threads=3)
    res = {}
    for r in rules:
        aucs = {}
        for name, _, te in windows(r):
            aucs[name] = roc_auc_score(c.y[te], ensemble_scores([load(task_path(r, name, m)) for m in MEMBERS]))
        res[r] = {'auc': float(np.mean(list(aucs.values()))), 'by_window': aucs}
        print(r, '%.4f' % res[r]['auc'], flush=True)
    json.dump(res, open(os.path.join(OUT, 'diagnostic_splits.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()

"""Leakage audit of the adopted PROTOCOL_AUC.md candidate, validation folds only.

    python experiments/audit_auc.py

1. First-time frauds: the gain on frauds whose firm has no fraud label left in training.
2. Placebo: training labels shuffled within each fiscal year must give AUC near 0.5.
3. Market block shuffled within each fiscal year: the gain must fall back to what the rest gives alone.
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from common import N_BAGS, OUT, ctx, keys, load, mask_serial, os, path, run_tasks, split
from models import TREE_LGBM, TREE_XGB, LogReg, WeightedBagger, ensemble_scores, lgbm_factory, recency_weights, xgb_factory
from run_auc import CHECK_SEEDS, SETTING, frame, reference, score, tt_score
from run_reference import MEMBERS


def fit_variant(key, member, seed, mode, n_jobs):
    """Refit one member with shuffled labels ('labels') or a shuffled market block ('market')."""
    c = ctx()
    tr, te = split(key)
    ytr = mask_serial(c.y, c.paaer, tr, te)
    D, feats = frame(tr)
    rng = np.random.RandomState(seed + 500)
    if mode == 'labels':
        for yr in np.unique(c.years[tr]):
            rows = np.where(c.years[tr] == yr)[0]
            ytr[rows] = rng.permutation(ytr[rows])
    else:
        cols = [f for f in feats if f.startswith(('mkt_', 'shares_'))]
        vals = D[cols].values.copy()
        for yr in np.unique(c.years):
            rows = np.where(c.years == yr)[0]
            vals[rows] = vals[rng.permutation(rows)]
        D[cols] = vals
    X = D[feats].values
    if member == 'LR':
        return LogReg().fit(X[tr], ytr).predict_proba(X[te])
    p = TREE_LGBM if member == 'LGBM' else TREE_XGB
    make, sd = (lgbm_factory(p, n_jobs), seed) if member == 'LGBM' else (xgb_factory(p, n_jobs), seed + 1)
    return WeightedBagger(make, N_BAGS, p['ratio'], sd).fit(X[tr], ytr, recency_weights(c.years[tr])).predict_proba(X[te])


def variant_path(key, member, seed, mode):
    return path('auc', 'audit', f'{mode}_{SETTING}_{key}_{member}_s{seed}.pkl')


def first_time():
    c = ctx()
    out = {'reference': [], 'candidate': []}
    n, new = 0, 0
    for k in keys('cv'):
        tr, te = split(k)
        ytr = mask_serial(c.y, c.paaer, tr, te)
        known = np.isin(c.gvkey[te], c.gvkey[tr][ytr == 1])
        y = c.y[te]
        keep = ~((y == 1) & known)
        n, new = n + int(y.sum()), new + int(y[keep].sum())
        for s in CHECK_SEEDS:
            out['reference'].append(roc_auc_score(y[keep], reference(k, s)[keep]))
            out['candidate'].append(roc_auc_score(y[keep], score(k, s)[keep]))
    return {'frauds': n, 'first_time_frauds': new, 'reference': float(np.mean(out['reference'])), 'candidate': float(np.mean(out['candidate']))}


def main():
    c = ctx()
    res = {'first_time': first_time()}
    print(res['first_time'], flush=True)
    run_tasks([(fit_variant, variant_path(k, m, sd, mode), dict(key=k, member=m, seed=sd, mode=mode))
               for mode in ('labels', 'market') for k in keys('cv') for m in MEMBERS for sd in CHECK_SEEDS], n_workers=5, threads=3)

    def auc(fn):
        return float(np.mean([roc_auc_score(c.y[split(k)[1]], fn(k, sd)) for k in keys('cv') for sd in CHECK_SEEDS]))

    res['placebo_trees_only'] = auc(lambda k, sd: ensemble_scores([load(variant_path(k, m, sd, 'labels')) for m in MEMBERS]))
    print('placebo, shuffled labels, trees and LR only:', res['placebo_trees_only'], flush=True)
    ref = auc(reference)
    res['gain'] = auc(score) - ref
    res['gain_market_shuffled'] = auc(lambda k, sd: ensemble_scores([load(variant_path(k, m, sd, 'market')) for m in MEMBERS] + [tt_score(k, sd)])) - ref
    print('gain', res['gain'], 'with market block shuffled', res['gain_market_shuffled'], flush=True)
    json.dump(res, open(os.path.join(OUT, 'audit_auc.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()

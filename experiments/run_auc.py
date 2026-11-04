"""The adopted candidate of PROTOCOL_AUC.md (market features, recency weights, retuned trees) against the
previous ensemble. The other families and the filing data of PROTOCOL_TEXT.md did not pass: their code is
in git history at f7c0284 and their scores in results/experiments/.

    python experiments/run_auc.py --stage retune      # how models.TREE_LGBM and TREE_XGB were chosen
    python experiments/run_auc.py --stage cv          # validation years, fresh seeds
    python experiments/run_auc.py --stage test        # test years, then experiments/compare_auc.py
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import N_BAGS, OUT, SEEDS, ctx, keys, load, mask_serial, os, path, pickle, run_tasks, split, winsorized
from features import market_features
from models import (TREE_LGBM, TREE_XGB, BalancedBagger, LogReg, WeightedBagger, ensemble_scores, lgbm_factory,
                    recency_weights, sample_lgbm, sample_xgb, xgb_factory)
from run_deep import cv_path, final_path
from run_reference import MEMBERS, ref_path
from scoring import cv_report

SETTING = 'A1_mkt+A4_3+A11_both'
SCREEN_SEED, CHECK_SEEDS = 42, (43, 44)
N_RETUNE, TUNE_BAGS = 40, 3


def market():
    """Earlier years and same-year ranks only, so built once for the panel."""
    p = path('auc', 'market.pkl')
    if not os.path.exists(p):
        c = ctx()
        with open(p, 'wb') as fh:
            pickle.dump(market_features(c.DW, c.gvkey, c.years), fh)
    return load(p)


def frame(tr):
    """Window's winsorised features plus the market block, and the feature names to use."""
    c = ctx()
    D = winsorized(tr)
    extra = market()
    extra.index = D.index
    return pd.concat([D, extra], axis=1), list(c.feats) + list(extra.columns)


def member_task(key, member, seed, n_jobs):
    c = ctx()
    tr, te = split(key)
    ytr = mask_serial(c.y, c.paaer, tr, te)
    D, feats = frame(tr)
    X = D[feats].values
    if member == 'LR':
        return LogReg().fit(X[tr], ytr).predict_proba(X[te])
    p = TREE_LGBM if member == 'LGBM' else TREE_XGB
    make, sd = (lgbm_factory(p, n_jobs), seed) if member == 'LGBM' else (xgb_factory(p, n_jobs), seed + 1)
    return WeightedBagger(make, N_BAGS, p['ratio'], sd).fit(X[tr], ytr, recency_weights(c.years[tr])).predict_proba(X[te])


def retune_task(member, trial, key, n_jobs):
    c = ctx()
    tr, te = split(key)
    ytr = mask_serial(c.y, c.paaer, tr, te)
    X = winsorized(tr)[c.feats].values
    p = trial_params(member, trial)
    make, sd = (lgbm_factory(p, n_jobs), SCREEN_SEED) if member == 'LGBM' else (xgb_factory(p, n_jobs), SCREEN_SEED + 1)
    return BalancedBagger(make, TUNE_BAGS, p['ratio'], sd).fit(X[tr], ytr).predict_proba(X[te])


def trial_params(member, trial):
    rng = np.random.RandomState({'LGBM': 2000, 'XGB': 2001}[member])
    sampler = sample_lgbm if member == 'LGBM' else sample_xgb
    return [sampler(rng) for _ in range(trial + 1)][trial]


def retune_path(member, trial, key):
    return path('auc', 'retune', f'{member}_t{trial}_{key}.pkl')


def member_path(key, member, seed):
    return path('auc', 'member', f'{SETTING}_{key}_{member}_s{seed}.pkl')


def tt_score(key, seed):
    return load(cv_path('temp_trans', 0, key, seed))[15] if key.startswith('val') and seed == SCREEN_SEED else load(final_path('temp_trans', key, seed))


def reference(key, seed):
    return ensemble_scores([load(ref_path(key, m, seed)) for m in MEMBERS] + [tt_score(key, seed)])


def score(key, seed):
    return ensemble_scores([load(member_path(key, m, seed)) for m in MEMBERS] + [tt_score(key, seed)])


def tasks(stage_keys, seeds):
    return [(member_task, member_path(k, m, s), dict(key=k, member=m, seed=s)) for k in stage_keys for s in seeds for m in MEMBERS]


def stage_retune():
    """Best trial per tree model: the one that most helps the full ensemble on the validation years."""
    c = ctx()
    run_tasks([(retune_task, retune_path(m, t, k), dict(member=m, trial=t, key=k))
               for m in ('LGBM', 'XGB') for t in range(N_RETUNE) for k in keys('cv')], n_workers=5, threads=3)
    table = {}
    for m in ('LGBM', 'XGB'):
        aucs = []
        for t in range(N_RETUNE):
            per = []
            for k in keys('cv'):
                others = [load(ref_path(k, o, SCREEN_SEED)) for o in MEMBERS if o != m] + [tt_score(k, SCREEN_SEED)]
                per.append(roc_auc_score(c.y[split(k)[1]], ensemble_scores(others + [load(retune_path(m, t, k))])))
            aucs.append(float(np.mean(per)))
        t = int(np.argmax(aucs))
        table[m] = {'best_trial': t, 'auc': aucs[t], 'params': trial_params(m, t)}
    print(json.dumps(table, indent=2), flush=True)
    assert table['LGBM']['params'] == TREE_LGBM and table['XGB']['params'] == TREE_XGB, 'differs from models.py'


def stage_cv():
    run_tasks(tasks(keys('cv'), CHECK_SEEDS), n_workers=5, threads=3)
    ref = cv_report({s: {k: reference(k, s) for k in keys('cv')} for s in CHECK_SEEDS})
    new = cv_report({s: {k: score(k, s) for k in keys('cv')} for s in CHECK_SEEDS})
    print('reference', ref, '\ncandidate', new, '\ngain %+.4f' % (new['cv_auc'] - ref['cv_auc']), flush=True)


def stage_test():
    run_tasks(tasks(keys('test'), SEEDS), n_workers=5, threads=3)
    for s in SEEDS:
        with open(path('auc', 'candidates', f'{SETTING}_s{s}.pkl'), 'wb') as fh:
            pickle.dump({k: score(k, s) for k in keys('test')}, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['retune', 'cv', 'test'], required=True)
    a = ap.parse_args()
    ctx()
    market()
    {'retune': stage_retune, 'cv': stage_cv, 'test': stage_test}[a.stage]()


if __name__ == '__main__':
    main()

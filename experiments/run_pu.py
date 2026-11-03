"""Focal PU XGBoost (iFPU): prior and gamma chosen on pre-2003 folds, either replacing XGB or joining.

    python experiments/run_pu.py --stage cv
    python experiments/run_pu.py --stage test --pi 0.05 --gamma 3 --mode replace
"""
import argparse
import json

from common import OUT, SEEDS, ctx, keys, load, mask_serial, os, path, run_tasks, split, winsorized
from models import ensemble_scores
from pu_learning import PUXGB
from run_reference import ref_path
from scoring import cv_report, reference_cv, save_candidate

PIS = (0.01, 0.02, 0.05, 0.10)
GAMMAS = (1.0, 3.0)
MODES = {'replace': ('LGBM', 'LR'), 'add': ('LGBM', 'XGB', 'LR')}
CV_SEEDS = (42,)


def task(key, pi, gamma, seed, n_jobs):
    c = ctx()
    tr, te = split(key)
    ytr = mask_serial(c.y, c.paaer, tr, te)
    X = winsorized(tr)[c.feats].values
    return PUXGB(c.bestX, pi=pi, gamma=gamma, seed=seed, n_jobs=n_jobs).fit(X[tr], ytr).predict_proba(X[te])


def pu_path(key, pi, gamma, seed):
    return path('pu', f'{key}_pi{pi}_g{gamma}_s{seed}.pkl')


def ens(key, pi, gamma, mode, seed):
    return ensemble_scores([load(ref_path(key, m, seed)) for m in MODES[mode]] + [load(pu_path(key, pi, gamma, seed))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['cv', 'test'], required=True)
    ap.add_argument('--pi', type=float)
    ap.add_argument('--gamma', type=float)
    ap.add_argument('--mode', choices=list(MODES))
    a = ap.parse_args()
    ctx()
    if a.stage == 'cv':
        run_tasks([(task, pu_path(k, p, g, s), dict(key=k, pi=p, gamma=g, seed=s))
                   for s in CV_SEEDS for p in PIS for g in GAMMAS for k in keys('cv')], n_workers=4, threads=4)
        rows = []
        for p in PIS:
            for g in GAMMAS:
                alone = cv_report({s: {k: load(pu_path(k, p, g, s)) for k in keys('cv')} for s in CV_SEEDS})
                rows.append({'pi': p, 'gamma': g, 'mode': 'alone', **alone})
                for mode in MODES:
                    rows.append({'pi': p, 'gamma': g, 'mode': mode,
                                 **cv_report({s: {k: ens(k, p, g, mode, s) for k in keys('cv')} for s in CV_SEEDS})})
        ref = reference_cv(CV_SEEDS)
        json.dump({'reference': ref, 'candidates': rows}, open(os.path.join(OUT, 'cv_pu.json'), 'w'), indent=2)
        print('reference', ref)
        for r in sorted(rows, key=lambda r: -r['cv_auc']):
            print(r)
    else:
        run_tasks([(task, pu_path(k, a.pi, a.gamma, s), dict(key=k, pi=a.pi, gamma=a.gamma, seed=s))
                   for s in SEEDS for k in keys('test')], n_workers=4, threads=4)
        for s in SEEDS:
            save_candidate(f'PU_pi{a.pi}_g{a.gamma}_{a.mode}', s, {k: ens(k, a.pi, a.gamma, a.mode, s) for k in keys('test')})


if __name__ == '__main__':
    main()

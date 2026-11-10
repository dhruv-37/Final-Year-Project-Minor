"""Reference model scores: LGBM, XGB, LR on 133 features, for CV years and test years, three seeds.

    python experiments/run_reference.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import N_BAGS, ROOT, SEEDS, ctx, fit_member, keys, load, mask_serial, os, path, run_tasks, split, winsorized

MEMBERS = ('LGBM', 'XGB', 'LR')


def task(key, member, seed, n_jobs):
    c = ctx()
    tr, te = split(key)
    ytr = mask_serial(c.y, c.paaer, tr, te)
    return fit_member(member, winsorized(tr), c.feats, tr, te, ytr, seed, N_BAGS, n_jobs)


def ref_path(key, member, seed):
    return path('reference', f'{key}_{member}_s{seed}.pkl')


def main():
    ctx()
    tasks = [(task, ref_path(k, m, s), dict(key=k, member=m, seed=s))
             for s in SEEDS for stage in ('cv', 'test') for k in keys(stage) for m in MEMBERS]
    run_tasks(tasks, n_workers=4, threads=4)

    # same code path as run_pipeline.py? seed 42 must reproduce the committed predictions
    saved = pd.read_csv(os.path.join(ROOT, 'results', 'test_predictions.csv'))
    c = ctx()
    col = {'LGBM': 'LightGBM bagged (enhanced)', 'XGB': 'XGBoost bagged (enhanced)', 'LR': 'Logistic regression (enhanced)'}
    for k in keys('test'):
        _, te = split(k)
        s = saved[saved.fyear == int(k[-4:])].sort_values('gvkey')
        order = np.argsort(c.gvkey[te], kind='stable')
        for m in MEMBERS:
            p = load(ref_path(k, m, 42))[order]
            print(k, m, 'max abs diff from committed', f'{np.abs(p - s[col[m]].values).max():.2e}',
                  'AUC', f'{roc_auc_score(c.y[te], p):.4f}', flush=True)


if __name__ == '__main__':
    main()

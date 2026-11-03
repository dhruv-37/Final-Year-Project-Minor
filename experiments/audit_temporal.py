"""Leakage audit of the temporal models, validation folds only.

    python experiments/audit_temporal.py

1. Structure: every history entry is the same firm, in consecutive earlier fiscal years.
2. Placebo: training labels shuffled row by row within each fiscal year must give AUC near 0.5.
3. First-time frauds: scores restricted to validation frauds whose firm has no visible fraud label
   in training. A gain that came from memorising repeat offenders would vanish here.
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from common import OUT, ctx, keys, load, mask_serial, os, path, run_tasks, split
from deep_models import DeepClassifier, history_index
from models import ensemble_scores
from run_deep import CHECK_SEEDS, HIST_LEN, final_path, inputs, sample_config
from run_reference import MEMBERS, ref_path

PLACEBO_SEEDS = (7, 8, 9)


def structure():
    c = ctx()
    idx = history_index(c.gvkey, c.years, HIST_LEN)
    row = np.arange(len(idx))
    bad = 0
    for j in range(HIST_LEN - 1):
        ok = idx[:, j] >= 0
        back = HIST_LEN - 1 - j
        bad += int((c.gvkey[idx[ok, j]] != c.gvkey[row[ok]]).sum() + (c.years[idx[ok, j]] != c.years[row[ok]] - back).sum())
    return {'rows': len(idx), 'violations': bad, 'mean_history_len': float((idx >= 0).sum(1).mean())}


def placebo_task(arch, config, epochs, key, seed, n_jobs):
    c = ctx()
    kw, train = config
    Itr, Ite, tr, te = inputs(arch, key)
    ytr = mask_serial(c.y, c.paaer, tr, te)
    rng = np.random.RandomState(seed)
    shuffled = ytr.copy()
    for yr in np.unique(c.years[tr]):
        rows = np.where(c.years[tr] == yr)[0]
        shuffled[rows] = rng.permutation(ytr[rows])
    return DeepClassifier(arch, kw, n_members=1, epochs=epochs, seed=seed, **train).fit_predict(Itr, shuffled, Ite)


def first_time(arch):
    """AUC on validation rows, dropping frauds whose firm has a fraud label left in training."""
    c = ctx()
    out = {}
    for name, score in (('reference', lambda k, s: ensemble_scores([load(ref_path(k, m, s)) for m in MEMBERS])),
                        (arch, lambda k, s: load(final_path(arch, k, s))),
                        ('reference+' + arch, lambda k, s: ensemble_scores([load(ref_path(k, m, s)) for m in MEMBERS] + [load(final_path(arch, k, s))]))):
        all_auc, new_auc, n_pos, n_new = [], [], 0, 0
        for k in keys('cv'):
            tr, te = split(k)
            ytr = mask_serial(c.y, c.paaer, tr, te)
            known = np.isin(c.gvkey[te], c.gvkey[tr][ytr == 1])
            y = c.y[te]
            keep = ~((y == 1) & known)
            for s in CHECK_SEEDS:
                p = score(k, s)
                all_auc.append(roc_auc_score(y, p))
                new_auc.append(roc_auc_score(y[keep], p[keep]))
            n_pos += int(y.sum())
            n_new += int(y[keep].sum())
        out[name] = {'all': float(np.mean(all_auc)), 'first_time': float(np.mean(new_auc)), 'frauds': n_pos, 'first_time_frauds': n_new}
    return out


def main():
    res = {'structure': structure()}
    print(res['structure'], flush=True)
    frozen = json.load(open(os.path.join(OUT, 'cv_deep.json')))
    archs = ['temp_trans', 'temp_mamba']
    run_tasks([(placebo_task, path('audit', f'placebo_row_{a}_{k}_s{s}.pkl'),
                dict(arch=a, config=sample_config(a, frozen[a]['frozen']['trial']), epochs=frozen[a]['frozen']['epochs'], key=k, seed=s))
               for a in archs for k in keys('cv') for s in PLACEBO_SEEDS], n_workers=5, threads=3)
    c = ctx()
    for a in archs:
        aucs = {k: [float(roc_auc_score(c.y[split(k)[1]], load(path('audit', f'placebo_row_{a}_{k}_s{s}.pkl')))) for s in PLACEBO_SEEDS]
                for k in keys('cv')}
        res[a] = {'placebo_auc': aucs, 'placebo_mean': float(np.mean(list(aucs.values()))), 'first_time': first_time(a)}
        print(a, json.dumps(res[a], indent=1), flush=True)
    json.dump(res, open(os.path.join(OUT, 'audit_temporal.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()

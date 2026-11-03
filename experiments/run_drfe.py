"""Dynamic recursive feature elimination, chosen on pre-2003 folds, then applied per test window.

    python experiments/run_drfe.py --stage cv
    python experiments/run_drfe.py --stage test --tau 0.05 --rule one_se
"""
import argparse
import json

import numpy as np

from common import N_BAGS, OUT, SEEDS, ctx, fit_member, keys, load, mask_serial, os, path, run_tasks, split, winsorized
from feature_selection import DynamicRFE, choose_subset
from models import ensemble_scores
from run_reference import MEMBERS
from scoring import cv_report, reference_cv, save_candidate

TAUS = (0.02, 0.05, 0.10)
RULES = ('best', 'one_se')
CV_SEEDS = (42, 43)


def path_task(key, tau, seed, n_jobs):
    c = ctx()
    tr, te = split(key)
    ytr = mask_serial(c.y, c.paaer, tr, te)
    X = winsorized(tr)[c.feats].values[tr]
    return DynamicRFE(c.bestL, tau=tau, seed=seed, n_jobs=n_jobs).fit(X, ytr, c.years[tr], c.paaer[tr], c.feats).path_


def member_task(key, tau, rule, member, seed, n_jobs):
    c = ctx()
    tr, te = split(key)
    ytr = mask_serial(c.y, c.paaer, tr, te)
    subset = choose_subset(load(p_path(key, tau, seed)), rule)
    return fit_member(member, winsorized(tr), subset, tr, te, ytr, seed, N_BAGS, n_jobs)


def p_path(key, tau, seed):
    return path('drfe', f'path_{key}_tau{tau}_s{seed}.pkl')


def m_path(key, tau, rule, member, seed):
    return path('drfe', f'{key}_tau{tau}_{rule}_{member}_s{seed}.pkl')


def run(stage, taus, rules, seeds):
    run_tasks([(path_task, p_path(k, t, s), dict(key=k, tau=t, seed=s))
               for s in seeds for t in taus for k in keys(stage)], n_workers=5, threads=3)
    run_tasks([(member_task, m_path(k, t, r, m, s), dict(key=k, tau=t, rule=r, member=m, seed=s))
               for s in seeds for t in taus for r in rules for k in keys(stage) for m in MEMBERS],
              n_workers=4, threads=4)


def ensemble(key, tau, rule, seed):
    return ensemble_scores([load(m_path(key, tau, rule, m, seed)) for m in MEMBERS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['cv', 'test'], required=True)
    ap.add_argument('--tau', type=float)
    ap.add_argument('--rule')
    a = ap.parse_args()
    ctx()
    if a.stage == 'cv':
        run('cv', TAUS, RULES, CV_SEEDS)
        rows = []
        for t in TAUS:
            for r in RULES:
                n = np.mean([len(choose_subset(load(p_path(k, t, s)), r)) for k in keys('cv') for s in CV_SEEDS])
                scores = {s: {k: ensemble(k, t, r, s) for k in keys('cv')} for s in CV_SEEDS}
                rows.append({'tau': t, 'rule': r, 'mean_n_features': float(n), **cv_report(scores)})
        ref = reference_cv()
        os.makedirs(OUT, exist_ok=True)
        json.dump({'reference': ref, 'candidates': rows}, open(os.path.join(OUT, 'cv_drfe.json'), 'w'), indent=2)
        print('reference', ref)
        for r in sorted(rows, key=lambda r: -r['cv_auc']):
            print(r)
    else:
        run('test', [a.tau], [a.rule], SEEDS)
        for s in SEEDS:
            save_candidate(f'DRFE_tau{a.tau}_{a.rule}', s, {k: ensemble(k, a.tau, a.rule, s) for k in keys('test')})
        sel = {k: {f's{s}': choose_subset(load(p_path(k, a.tau, s)), a.rule) for s in SEEDS} for k in keys('test')}
        json.dump(sel, open(os.path.join(OUT, 'drfe_selected_features.json'), 'w'), indent=1)
        for k, v in sel.items():
            print(k, {s: len(f) for s, f in v.items()}, flush=True)


if __name__ == '__main__':
    main()

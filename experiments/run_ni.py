"""Protocol 6 (PROTOCOL_NI.md): is dynamic feature selection no worse than the fixed pool?

    python experiments/run_ni.py --stage cv        # validation years, seeds 43 and 44: safety check
    python experiments/run_ni.py --stage test      # test years, 3 seeds, only if the check passed
    python experiments/run_ni.py --stage compare   # bootstrap, verdict against the margin
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import N_BAGS, OUT, SEEDS, TEST_YEARS, ctx, keys, load, mask_serial, os, path, pickle, run_tasks, split
from feature_selection import DynamicRFE, choose_subset
from models import TREE_LGBM, TREE_XGB, LogReg, WeightedBagger, ensemble_scores, lgbm_factory, recency_weights, xgb_factory
from run_auc import frame, market
from run_auc import score as reference
from run_auc import tt_score
from run_reference import MEMBERS
from scoring import cv_report

TAU, RULE = 0.02, 'best'                        # frozen in PROTOCOL_FS.md (F9_0.02_best)
MARGIN = 0.003
CHECK_SEEDS = (43, 44)
HIST_BASES = [b + '_yr' for b in ('soft_assets', 'dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'ch_cs', 'ch_roa', 'bm', 'reoa',
                                  'EBIT', 'roa', 'margin', 'asset_turn', 'lev', 'cash_ta', 'rect_sale', 'inv_cogs',
                                  'mve_ta', 'fin_need', 'tata')]


def history_features(frame_, gvkey, years, bases, window=4):
    """As in PROTOCOL_FS.md: for each base column, the deviation from the firm's own mean over its previous
    `window` years, that deviation over the firm's own volatility, and the volatility. Only earlier years
    enter, and a gap in fiscal years starts a new history. Rows sorted by firm, then year."""
    spell = np.cumsum(np.r_[True, (gvkey[1:] != gvkey[:-1]) | (years[1:] != years[:-1] + 1)])
    now = frame_[bases].reset_index(drop=True)
    past = now.groupby(spell).shift(1)
    roll = past.groupby(spell).rolling(window, min_periods=2)
    mean = roll.mean().reset_index(level=0, drop=True).sort_index()
    vol = roll.std().reset_index(level=0, drop=True).sort_index()
    dev = now - mean
    z = (dev / vol.where(vol > 1e-3)).clip(-5, 5)
    return pd.concat([dev.add_suffix('_dev'), z.add_suffix('_z'), vol.add_suffix('_vol')], axis=1).astype(np.float32)


def history():
    p = path('ni', 'history.pkl')
    if not os.path.exists(p):
        c = ctx()
        with open(p, 'wb') as fh:
            pickle.dump(history_features(c.DW, c.gvkey, c.years, HIST_BASES), fh)
    return load(p)


def pool(tr):
    """The window's 140 features and the 60 history features, with the pool's names."""
    D, feats = frame(tr)
    h = history()
    h.index = D.index
    return pd.concat([D, h], axis=1), feats + list(h.columns)


def select_task(key, seed, n_jobs):
    c = ctx()
    tr, te = split(key)
    ytr = mask_serial(c.y, c.paaer, tr, te)
    P, names = pool(tr)
    X = P[names].values[tr]
    path_ = DynamicRFE(c.bestL, tau=TAU, seed=seed, n_jobs=n_jobs).fit(X, ytr, c.years[tr], c.paaer[tr], names).path_
    return {'path': path_, 'chosen': choose_subset(path_, RULE)}


def select_path(key, seed):
    return path('ni', 'select', f'{key}_s{seed}.pkl')


def member_task(key, member, seed, n_jobs):
    c = ctx()
    tr, te = split(key)
    ytr = mask_serial(c.y, c.paaer, tr, te)
    P, _ = pool(tr)
    X = P[load(select_path(key, seed))['chosen']].values
    if member == 'LR':
        return LogReg().fit(X[tr], ytr).predict_proba(X[te])
    p = TREE_LGBM if member == 'LGBM' else TREE_XGB
    make, sd = (lgbm_factory(p, n_jobs), seed) if member == 'LGBM' else (xgb_factory(p, n_jobs), seed + 1)
    return WeightedBagger(make, N_BAGS, p['ratio'], sd).fit(X[tr], ytr, recency_weights(c.years[tr])).predict_proba(X[te])


def member_path(key, member, seed):
    return path('ni', 'member', f'{key}_{member}_s{seed}.pkl')


def candidate(key, seed):
    return ensemble_scores([load(member_path(key, m, seed)) for m in MEMBERS] + [tt_score(key, seed)])


def run(stage_keys, seeds):
    run_tasks([(select_task, select_path(k, s), dict(key=k, seed=s)) for k in stage_keys for s in seeds], n_workers=5, threads=3)
    run_tasks([(member_task, member_path(k, m, s), dict(key=k, member=m, seed=s)) for k in stage_keys for s in seeds for m in MEMBERS],
              n_workers=4, threads=4)


def n_chosen(stage_keys, seeds):
    sizes = {f'{k}_s{s}': len(load(select_path(k, s))['chosen']) for k in stage_keys for s in seeds}
    names = set(history().columns)              # by name: altman_z also ends in _z
    hist = [sum(f in names for f in load(select_path(k, s))['chosen']) for k in stage_keys for s in seeds]
    return {'by_window': sizes, 'mean': float(np.mean(list(sizes.values()))), 'mean_history_features': float(np.mean(hist))}


def stage_cv():
    run(keys('cv'), CHECK_SEEDS)
    ref = cv_report({s: {k: reference(k, s) for k in keys('cv')} for s in CHECK_SEEDS})
    new = cv_report({s: {k: candidate(k, s) for k in keys('cv')} for s in CHECK_SEEDS})
    out = {'reference': ref, 'candidate': new, 'change': new['cv_auc'] - ref['cv_auc'],
           'continue': bool(new['cv_auc'] - ref['cv_auc'] > -MARGIN), 'features': n_chosen(keys('cv'), CHECK_SEEDS)}
    json.dump(out, open(os.path.join(OUT, 'cv_ni.json'), 'w'), indent=2)
    print(json.dumps(out, indent=2), flush=True)


def stage_test():
    assert json.load(open(os.path.join(OUT, 'cv_ni.json')))['continue'], 'stopped by the validation check (PROTOCOL_NI.md)'
    run(keys('test'), SEEDS)


def stage_compare():
    from compare import MIN_FRAUDS, N_BOOT, PERIODS, stacked
    from evaluate import BOOT_METRICS, bootstrap_period, per_year_metrics, period_mean
    c = ctx()
    tk = keys('test')
    te = np.concatenate([split(k)[1] for k in tk])
    y, yrs = c.y[te], c.years[te]
    pos_key = np.where(np.isnan(c.paaer), -c.gvkey, c.paaer)[te]
    preds = {}
    for s in SEEDS:
        preds[('reference', s)] = stacked({k: reference(k, s) for k in tk}, tk)
        preds[('dynamic', s)] = stacked({k: candidate(k, s) for k in tk}, tk)
    yearly = {key: per_year_metrics(y, p, yrs) for key, p in preds.items()}
    use = [t for t in TEST_YEARS if y[yrs == t].sum() >= MIN_FRAUDS]
    boot = bootstrap_period(y, preds, yrs, pos_key, c.gvkey[te], use, n_boot=N_BOOT, seed=42)
    d = np.mean([boot[('dynamic', s)] - boot[('reference', s)] for s in SEEDS], axis=0)

    def mean(n, s, lo, hi):
        return period_mean(yearly[(n, s)], lo, hi, MIN_FRAUDS)

    rows = {}
    for j, met in enumerate(BOOT_METRICS):
        ref_v = float(np.mean([mean('reference', s, 2003, 2012)[met] for s in SEEDS]))
        new_v = float(np.mean([mean('dynamic', s, 2003, 2012)[met] for s in SEEDS]))
        rows[met] = {'reference': ref_v, 'dynamic': new_v, 'change': new_v - ref_v,
                     'ci95': [float(np.percentile(d[:, j], 2.5)), float(np.percentile(d[:, j], 97.5))],
                     'by_seed': [float(mean('dynamic', s, 2003, 2012)[met] - mean('reference', s, 2003, 2012)[met]) for s in SEEDS],
                     'by_period': {lbl: float(np.mean([mean('dynamic', s, lo, hi)[met] - mean('reference', s, lo, hi)[met] for s in SEEDS]))
                                   for lbl, (lo, hi) in PERIODS.items()}}
    for met in ('Prec@1%',):
        ref_v = float(np.mean([mean('reference', s, 2003, 2012)[met] for s in SEEDS]))
        new_v = float(np.mean([mean('dynamic', s, 2003, 2012)[met] for s in SEEDS]))
        rows[met] = {'reference': ref_v, 'dynamic': new_v, 'change': new_v - ref_v, 'ci95': None}
    lo, hi = rows['AUC']['ci95']
    verdict = ('non-inferior and superior' if lo > 0 else 'non-inferior' if lo > -MARGIN
               else 'measurable cost' if hi < 0 else 'inconclusive')
    out = {'margin': MARGIN, 'verdict': verdict, 'metrics': rows, 'features': n_chosen(tk, SEEDS)}
    json.dump(out, open(os.path.join(OUT, 'ni_test.json'), 'w'), indent=2)
    print(json.dumps(out, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['cv', 'test', 'compare'], required=True)
    a = ap.parse_args()
    ctx()
    market()
    history()
    {'cv': stage_cv, 'test': stage_test, 'compare': stage_compare}[a.stage]()


if __name__ == '__main__':
    main()

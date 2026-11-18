"""PROTOCOL_MDA.md: MD&A text features against the committed final ensemble.

    python experiments/mda/run_mda.py --stage prepare   # link, vocabulary, counts, features for the CV windows
    python experiments/mda/run_mda.py --stage screen    # M1, M2, M1+M2 on seed 42, validation years
    python experiments/mda/run_mda.py --stage check     # best setting: seeds 43-44, MD&A rows, placebo, new cases
    python experiments/mda/run_mda.py --stage test      # only if the check passed: test years, 3 seeds
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import N_BAGS, OUT, ROOT, SEEDS, ctx, keys, load, mask_serial, path, pickle, run_tasks, split  # noqa: E402
from models import TREE_LGBM, TREE_XGB, LogReg, WeightedBagger, ensemble_scores, lgbm_factory, recency_weights, xgb_factory  # noqa: E402
from run_auc import frame as base_frame, market, score as reference, tt_score  # noqa: E402
from run_reference import MEMBERS  # noqa: E402

import mda_features as mf  # noqa: E402

SCREEN_SEED, CHECK_SEEDS = 42, (43, 44)
GATE, PLACEBO_MAX = 0.002, 0.001
SETTINGS = ['M1', 'M2', 'M1+M2']


def frame(setting, placebo, key, tr):
    D, feats = base_frame(tr)
    extra = []
    if 'M1' in setting.split('+'):
        extra.append(mf.m1_block(key, placebo))
    if 'M2' in setting.split('+'):
        extra.append(mf.m2_block(placebo))
    for e in extra:
        e.index = D.index
    return pd.concat([D] + extra, axis=1), feats + [col for e in extra for col in e.columns]


def member_task(setting, placebo, key, member, seed, n_jobs):
    c = ctx()
    tr, te = split(key)
    ytr = mask_serial(c.y, c.paaer, tr, te)
    D, feats = frame(setting, placebo, key, tr)
    X = D[feats].values
    if member == 'LR':
        return LogReg().fit(X[tr], ytr).predict_proba(X[te])
    p = TREE_LGBM if member == 'LGBM' else TREE_XGB
    make, sd = (lgbm_factory(p, n_jobs), seed) if member == 'LGBM' else (xgb_factory(p, n_jobs), seed + 1)
    return WeightedBagger(make, N_BAGS, p['ratio'], sd).fit(X[tr], ytr, recency_weights(c.years[tr])).predict_proba(X[te])


def member_path(setting, placebo, key, member, seed):
    return path('mda', 'member', f"{'placebo_' if placebo else ''}{setting}_{key}_{member}_s{seed}.pkl")


def tasks(settings, stage_keys, seeds, placebo=False):
    return [(member_task, member_path(st, placebo, k, m, s), dict(setting=st, placebo=placebo, key=k, member=m, seed=s))
            for st in settings for k in stage_keys for s in seeds for m in MEMBERS]


def score(setting, key, seed, placebo=False):
    return ensemble_scores([load(member_path(setting, placebo, key, m, seed)) for m in MEMBERS] + [tt_score(key, seed)])


def new_case():
    """Fraud rows whose AAER case starts in their own fiscal year (Protocol 9 definition)."""
    c = ctx()
    d = pd.DataFrame({'p': c.paaer, 'y': c.y, 'fy': c.years})
    start = d[d.y == 1].groupby('p').fy.min()
    a = pd.read_csv(os.path.join(ROOT, 'data', 'external', 'fresh', 'jarfraud', 'AAER_firm_year.csv'))
    start = pd.concat([start, a.groupby('P_AAER').YEARA.min()], axis=1).min(axis=1)
    return (c.y == 1) & (d.fy.values == d.p.map(start).values)


def mean_auc(fn, seeds, rows='all'):
    """Mean over folds and seeds. rows: 'all', 'mda' (firm-years with a usable MD&A), 'new' (new-case frauds
    against all non-frauds)."""
    c = ctx()
    has, new = mf.usable(), new_case()
    vals = []
    for k in keys('cv'):
        te = split(k)[1]
        keep = {'all': np.ones(len(te), bool), 'mda': has[te], 'new': (c.y[te] == 0) | new[te]}[rows]
        for s in seeds:
            vals.append(roc_auc_score(c.y[te][keep], fn(k, s)[keep]))
    return float(np.mean(vals))


def coverage():
    c = ctx()
    has = mf.usable()
    out = {}
    for k in keys('cv'):
        te = split(k)[1]
        y, h = c.y[te], has[te]
        out[k] = {'mda_nonfraud': round(float(h[y == 0].mean()), 3), 'mda_fraud': round(float(h[y == 1].mean()), 3),
                  'flag_only_auc': round(float(roc_auc_score(y, h.astype(float))), 4)}
    yrs = pd.Series(has).groupby(c.years).mean().round(3)
    return {'cv_years': out, 'share_with_mda_by_fiscal_year': {int(k): float(v) for k, v in yrs.items()}}


def stage_prepare():
    s = mf.link()['stats']
    print('link', s, flush=True)
    mf.counts()
    mf.m2_docs()
    for k in keys('cv'):
        mf.m1_docs(k)
    cov = coverage()
    json.dump({'link': s, **cov}, open(os.path.join(OUT, 'mda_coverage.json'), 'w'), indent=2)
    print(json.dumps(cov['cv_years'], indent=1), flush=True)


def stage_screen():
    market()
    run_tasks(tasks(SETTINGS, keys('cv'), (SCREEN_SEED,)), n_workers=4, threads=4)
    ref = {r: mean_auc(reference, (SCREEN_SEED,), r) for r in ('all', 'mda', 'new')}
    rows = []
    for st in SETTINGS:
        fn = lambda k, s, st=st: score(st, k, s)
        rows.append({'setting': st, **{f'gain_{r}': mean_auc(fn, (SCREEN_SEED,), r) - ref[r] for r in ref}})
        print(rows[-1], flush=True)
    json.dump({'reference': ref, 'candidates': rows}, open(os.path.join(OUT, 'cv_mda_screen.json'), 'w'), indent=2)


def stage_check():
    scr = json.load(open(os.path.join(OUT, 'cv_mda_screen.json')))['candidates']
    st = max(scr, key=lambda r: r['gain_all'])['setting']
    market()
    run_tasks(tasks([st], keys('cv'), CHECK_SEEDS), n_workers=4, threads=4)
    run_tasks(tasks([st], keys('cv'), CHECK_SEEDS, placebo=True), n_workers=4, threads=4)
    ref = {r: mean_auc(reference, CHECK_SEEDS, r) for r in ('all', 'mda', 'new')}
    fn = lambda k, s: score(st, k, s)
    fp = lambda k, s: score(st, k, s, placebo=True)
    r = {'setting': st, **{f'gain_{x}': mean_auc(fn, CHECK_SEEDS, x) - ref[x] for x in ref},
         'placebo_gain': mean_auc(fp, CHECK_SEEDS) - ref['all']}
    r['gate'] = bool(r['gain_all'] >= GATE and r['gain_mda'] >= GATE and r['placebo_gain'] <= PLACEBO_MAX)
    print(json.dumps({'reference': ref, **r}, indent=2), flush=True)
    json.dump({'reference': ref, 'check': r, 'frozen': [st] if r['gate'] else []},
              open(os.path.join(OUT, 'cv_mda.json'), 'w'), indent=2)


def stage_test():
    res = json.load(open(os.path.join(OUT, 'cv_mda.json')))
    assert res['frozen'], 'the check did not pass: the test years are not used'
    st = res['frozen'][0]
    if 'M1' in st:
        for k in keys('test'):
            mf.m1_docs(k)
    market()
    run_tasks(tasks([st], keys('test'), SEEDS), n_workers=4, threads=4)
    for s in SEEDS:
        with open(path('mda', 'candidates', f'{st}_s{s}.pkl'), 'wb') as fh:
            pickle.dump({k: score(st, k, s) for k in keys('test')}, fh)


# ------------------------------------------------------------------ Protocol 10b: text-only member T
def text_docs(key, placebo=False):
    """Member-T score per MD&A (index order): L2 logistic regression on window TF-IDF, trained on training
    firm-years with text. Placebo: texts permuted among firm-years with text within each fiscal year (seed 7)."""
    p = path('mda', 'text', f"{'placebo_' if placebo else ''}{key}.pkl")
    if not os.path.exists(p):
        from sklearn.linear_model import LogisticRegression
        c = ctx()
        idx = mf.link()['index']
        rows, fy = idx['row'].values, idx['fyear'].values
        order = np.arange(len(idx))
        if placebo:
            rng = np.random.RandomState(7)
            for yr in np.unique(fy):
                r = np.where(fy == yr)[0]
                order[r] = r[rng.permutation(len(r))]
        tr, te = split(key)
        lab = np.full(len(c.y), -1)
        lab[tr] = mask_serial(c.y, c.paaer, tr, te)
        fit = np.where(np.isin(rows, tr))[0]
        T = mf._tfidf(mf.counts(), fit)[order]
        m = LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear', max_iter=1000)
        m.fit(T[fit], lab[rows[fit]])
        s = m.decision_function(T).astype(np.float32)
        with open(p, 'wb') as fh:
            pickle.dump(s, fh)
        print(f'  T {key}{" placebo" if placebo else ""}: {len(fit)} training MD&As, {int(lab[rows[fit]].sum())} frauds', flush=True)
    return load(p)


def text_member(key, placebo=False):
    """Panel-length member-T scores for the key's scored rows; rows without text take their donor's score."""
    return mf.to_panel(text_docs(key, placebo)[:, None])[:, 0][split(key)[1]]


def score_t(key, seed, placebo=False):
    from run_auc import member_path as final_member_path
    return ensemble_scores([load(final_member_path(key, m, seed)) for m in MEMBERS] + [tt_score(key, seed),
                           text_member(key, placebo)])


def stage_text():
    c = ctx()
    has = mf.usable()
    for k in keys('cv'):
        text_docs(k)
        text_docs(k, placebo=True)
    own = {}
    for k in keys('cv'):
        te = split(k)[1]
        h = has[te]
        own[k] = round(float(roc_auc_score(c.y[te][h], text_member(k)[h])), 4)
    rows = ('all', 'mda', 'new')
    out = {'member_T_auc_on_rows_with_text': own}
    for name, seeds in (('screen_seed42', (SCREEN_SEED,)), ('check_seeds43_44', CHECK_SEEDS)):
        ref = {r: mean_auc(reference, seeds, r) for r in rows}
        g = {f'gain_{r}': mean_auc(score_t, seeds, r) - ref[r] for r in rows}
        g['placebo_gain'] = mean_auc(lambda k, s: score_t(k, s, placebo=True), seeds) - ref['all']
        out[name] = {'reference': ref, **g}
        print(name, json.dumps(out[name], indent=1), flush=True)
    ck = out['check_seeds43_44']
    out['gate'] = bool(ck['gain_all'] >= GATE and ck['gain_mda'] >= GATE and ck['placebo_gain'] <= PLACEBO_MAX)
    print('member T AUC on rows with text', own, 'gate', out['gate'], flush=True)
    json.dump(out, open(os.path.join(OUT, 'cv_mda_text.json'), 'w'), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['prepare', 'screen', 'check', 'test', 'text'], required=True)
    a = ap.parse_args()
    ctx()
    {'prepare': stage_prepare, 'screen': stage_screen, 'check': stage_check, 'test': stage_test,
     'text': stage_text}[a.stage]()


if __name__ == '__main__':
    main()

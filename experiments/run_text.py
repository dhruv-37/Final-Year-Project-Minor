"""Filing-data candidates of PROTOCOL_TEXT.md against the current ensemble.

    python experiments/run_text.py --stage screen      # all settings, seed 42, validation years
    python experiments/run_text.py --stage check       # best settings: fresh seeds, linked rows, placebo
    python experiments/run_text.py --stage test --cand T1 ...
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import N_BAGS, OUT, ROOT, SEEDS, ctx, keys, load, mask_serial, os, path, pickle, run_tasks, split, winsorized
from auc_features import KEY_RANKS
from filing_features import build_blocks, load_filings, load_links, shuffled_block
from models import BalancedBagger, LogReg, ensemble_scores, lgbm_factory, xgb_factory
from run_auc import reference, tt_score
from run_reference import MEMBERS, ref_path

LM_CSV = os.path.join(ROOT, 'data', 'external', 'lm10x', 'Loughran-McDonald_10X_Summaries_1993-2025.csv')
LINKS_CSV = os.path.join(ROOT, 'data', 'external', 'farr', 'gvkey_ciks.csv')
SCREEN_SEED, CHECK_SEEDS = 42, (43, 44)
GATE, PLACEBO_MAX = 0.002, 0.001
SINGLE = ['T1', 'T2', 'T2n', 'T2p', 'T3', 'T4']
COMBOS = ['T1+T2', 'T1+T2+T3', 'T1+T2+T3+T4', 'T1+T2n', 'T1+T2p', 'T1+T2p+T3+T4']
PRIOR = 0.0067       # shrinks a small industry's fraud rate towards the panel's rate


def blocks():
    p = path('text', 'blocks.pkl')
    if not os.path.exists(p):
        c = ctx()
        b, linked = build_blocks(load_filings(LM_CSV), load_links(LINKS_CSV), c.gvkey, c.years, c.DW[KEY_RANKS])
        with open(p, 'wb') as fh:
            pickle.dump({'blocks': b, 'linked': linked}, fh)
    return load(p)


def placebo_blocks():
    p = path('text', 'placebo.pkl')
    if not os.path.exists(p):
        c = ctx()
        b = blocks()
        with open(p, 'wb') as fh:
            pickle.dump({k: shuffled_block(v, c.years, b['linked'], seed=7 + i) for i, (k, v) in enumerate(b['blocks'].items())}, fh)
    return load(p)


def _shrunk_rate(ff, y):
    g = pd.DataFrame({'ff': ff, 'y': y}).groupby('ff')['y'].agg(['sum', 'count'])
    return (g['sum'] + PRIOR * 100) / (g['count'] + 100)


def industry_rate(ff, tr, ytr):
    return ff.map(_shrunk_rate(ff.values[tr], ytr)).fillna(PRIOR).astype(np.float32)


def past_industry_rate(ff, tr, ytr):
    """Each row sees only labels at least 2 fiscal years older than itself, so no row sees its own."""
    c = ctx()
    out = np.full(len(ff), PRIOR, dtype=np.float32)
    for yr in np.unique(c.years):
        past = c.years[tr] <= yr - 2
        if past.any():
            rows = np.where(c.years == yr)[0]
            out[rows] = ff.iloc[rows].map(_shrunk_rate(ff.values[tr][past], ytr[past])).fillna(PRIOR).astype(np.float32).values
    return pd.Series(out, index=ff.index)


def frame(setting, placebo, tr, ytr):
    """Window's features plus the setting's blocks. The T2 industry rate uses training labels only."""
    c = ctx()
    D = winsorized(tr)
    source = placebo_blocks() if placebo else blocks()['blocks']
    extra = []
    for fam in setting.split('+'):
        b = source[fam[:2]].copy()
        if fam == 'T2':
            b['ff48_rate'] = industry_rate(b['ff48'], tr, ytr)
        elif fam == 'T2p':
            b['ff48_rate'] = past_industry_rate(b['ff48'], tr, ytr)
        if fam[:2] == 'T2':
            b = b.drop(columns='ff48')
        b.index = D.index
        extra.append(b)
    out = pd.concat([D] + extra, axis=1)
    return out, list(c.feats) + [col for e in extra for col in e.columns]


def member_task(setting, placebo, key, member, seed, n_jobs):
    c = ctx()
    tr, te = split(key)
    ytr = mask_serial(c.y, c.paaer, tr, te)
    D, feats = frame(setting, placebo, tr, ytr)
    X = D[feats].values
    if member == 'LR':
        return LogReg().fit(X[tr], ytr).predict_proba(X[te])
    p = c.bestL if member == 'LGBM' else c.bestX
    make, sd = (lgbm_factory(p, n_jobs), seed) if member == 'LGBM' else (xgb_factory(p, n_jobs), seed + 1)
    return BalancedBagger(make, N_BAGS, p['ratio'], sd).fit(X[tr], ytr).predict_proba(X[te])


def member_path(setting, placebo, key, member, seed):
    return path('text', 'member', f"{'placebo_' if placebo else ''}{setting}_{key}_{member}_s{seed}.pkl")


def tasks(settings, stage_keys, seeds, placebo=False):
    return [(member_task, member_path(st, placebo, k, m, s), dict(setting=st, placebo=placebo, key=k, member=m, seed=s))
            for st in settings for k in stage_keys for s in seeds for m in MEMBERS]


def score(setting, key, seed, placebo=False):
    return ensemble_scores([load(member_path(setting, placebo, key, m, seed)) for m in MEMBERS] + [tt_score(key, seed)])


def mean_auc(fn, seeds, linked_only=False):
    c = ctx()
    linked = blocks()['linked']
    vals = []
    for k in keys('cv'):
        te = split(k)[1]
        keep = linked[te] if linked_only else np.ones(len(te), dtype=bool)
        for s in seeds:
            vals.append(roc_auc_score(c.y[te][keep], fn(k, s)[keep]))
    return float(np.mean(vals))


def coverage():
    c = ctx()
    linked = blocks()['linked']
    rows = {}
    for k in keys('cv'):
        te = split(k)[1]
        y, l = c.y[te], linked[te]
        rows[k] = {'linked_nonfraud': float(l[y == 0].mean()), 'linked_fraud': float(l[y == 1].mean()),
                   'flag_only_auc': float(roc_auc_score(y, l.astype(float)))}
    return rows


def stage_screen():
    blocks()
    print('coverage', coverage(), flush=True)
    run_tasks(tasks(SINGLE + COMBOS, keys('cv'), (SCREEN_SEED,)), n_workers=5, threads=3)
    ref = mean_auc(reference, (SCREEN_SEED,))
    ref_linked = mean_auc(reference, (SCREEN_SEED,), True)
    rows = []
    for st in SINGLE + COMBOS:
        fn = lambda k, s, st=st: score(st, k, s)
        rows.append({'setting': st, 'gain': mean_auc(fn, (SCREEN_SEED,)) - ref, 'gain_linked': mean_auc(fn, (SCREEN_SEED,), True) - ref_linked})
    json.dump({'reference': ref, 'reference_linked': ref_linked, 'coverage': coverage(), 'candidates': rows},
              open(os.path.join(OUT, 'cv_text_screen.json'), 'w'), indent=2)
    print('reference', ref, 'linked', ref_linked, flush=True)
    for r in sorted(rows, key=lambda r: -r['gain']):
        print(r['setting'], 'gain %+.4f' % r['gain'], 'linked %+.4f' % r['gain_linked'], flush=True)


def stage_check():
    scr = json.load(open(os.path.join(OUT, 'cv_text_screen.json')))['candidates']
    by = {r['setting']: r['gain'] for r in scr}
    settings = ['T1', max(['T2', 'T2n', 'T2p'], key=by.get), 'T3', 'T4', max(COMBOS, key=by.get)]
    placebo_blocks()     # built here, because workers writing it at once read each other's partial file
    run_tasks(tasks(settings, keys('cv'), CHECK_SEEDS), n_workers=5, threads=3)
    run_tasks(tasks(settings, keys('cv'), CHECK_SEEDS, placebo=True), n_workers=5, threads=3)
    ref, ref_linked = mean_auc(reference, CHECK_SEEDS), mean_auc(reference, CHECK_SEEDS, True)
    rows = []
    for st in settings:
        fn = lambda k, s, st=st: score(st, k, s)
        fp = lambda k, s, st=st: score(st, k, s, placebo=True)
        r = {'setting': st, 'screen_gain': by[st], 'gain': mean_auc(fn, CHECK_SEEDS) - ref, 'gain_linked': mean_auc(fn, CHECK_SEEDS, True) - ref_linked,
             'placebo_gain': mean_auc(fp, CHECK_SEEDS) - ref}
        r['gate'] = bool(r['gain'] >= GATE and r['gain_linked'] >= GATE and r['placebo_gain'] <= PLACEBO_MAX)
        rows.append(r)
        print(r, flush=True)
    rows.sort(key=lambda r: -r['gain'])
    frozen = [r['setting'] for r in rows if r['gate']][:3]
    json.dump({'reference': ref, 'reference_linked': ref_linked, 'candidates': rows, 'frozen': frozen},
              open(os.path.join(OUT, 'cv_text.json'), 'w'), indent=2)
    print('frozen for test years:', frozen, flush=True)


def stage_test(cands):
    res = json.load(open(os.path.join(OUT, 'cv_text.json')))
    assert set(cands) <= set(res['frozen']), f'only frozen candidates may see the test years: {res["frozen"]}'
    for st in cands:
        run_tasks(tasks([st], keys('test'), SEEDS), n_workers=5, threads=3)
        for s in SEEDS:
            with open(path('text', 'candidates', f'{st}_s{s}.pkl'), 'wb') as fh:
                pickle.dump({k: score(st, k, s) for k in keys('test')}, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['screen', 'check', 'test'], required=True)
    ap.add_argument('--cand', nargs='*', default=[])
    a = ap.parse_args()
    ctx()
    if a.stage == 'screen':
        stage_screen()
    elif a.stage == 'check':
        stage_check()
    else:
        stage_test(a.cand)


if __name__ == '__main__':
    main()

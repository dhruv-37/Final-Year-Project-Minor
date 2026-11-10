"""Protocol 5 (PROTOCOL_FG.md): features generated and selected inside every training window.

    python experiments/run_fg.py --stage screen     # validation years, seed 42, every setting
    python experiments/run_fg.py --stage check      # validation years, fresh seeds 43 and 44, best per family
    python experiments/run_fg.py --stage test       # test years, only settings that passed the check
    python experiments/run_fg.py --stage compare    # test-year comparison against the reference
"""
import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from common import N_BAGS, OUT, SEEDS, ctx, keys, load, mask_serial, os, path, run_tasks, split
from features import RAW28, sd
from models import TREE_LGBM, TREE_XGB, LogReg, WeightedBagger, ensemble_scores, lgbm_factory, recency_weights, xgb_factory
from run_auc import frame, market
from run_auc import score as reference
from run_auc import tasks as reference_tasks
from run_reference import MEMBERS
from scoring import cv_report

FAMILIES = {'G1': 1, 'G2': 2}
KS = (10, 25, 50)
SCREEN_SEED, CHECK_SEEDS = 42, (43, 44)
GATE = 0.002
RATIO = 20                    # non-frauds per fraud, as the members
RUNG1_NEG, RUNG1_KEEP, SURVIVORS, ROUND2_INPUTS = 0.25, 0.20, 100, 25
BOOST = dict(n_estimators=30, learning_rate=0.1, num_leaves=7, min_child_samples=50)
CHUNK = 100


# ----------------------------------------------------------------------------- candidate features
def round1_specs():
    r = [f'r:{a}/{b}' for a in RAW28 for b in RAW28 if a != b]
    d = [f'd:{a}-{b}' for i, a in enumerate(RAW28) for b in RAW28[i + 1:]]
    return r + d


def round2_specs(inputs):
    return ([f'c:{s}' for s in inputs] + [f'm:{s}' for s in inputs]
            + [f'q:{a}|{b}' for a in inputs for b in inputs if a != b])


class Panel:
    """Raw items and each firm's unbroken run of years; rows sorted by firm, then year."""

    def __init__(self):
        c = ctx()
        self.R = c.DW[RAW28].astype(np.float64).reset_index(drop=True)
        self.years = c.years
        g, y = c.gvkey, c.years
        self.spell = np.cumsum(np.r_[True, (g[1:] != g[:-1]) | (y[1:] != y[:-1] + 1)])

    def value(self, spec):
        kind, body = spec.split(':', 1)
        if kind == 'r':
            a, b = body.split('/')
            v = sd(self.R[a], self.R[b])
        elif kind == 'd':
            a, b = body.split('-')
            v = sd(self.R[a] - self.R[b], self.R['at'])
        elif kind == 'c':
            base = self.value(body)
            v = base - base.groupby(self.spell).shift(1)
        elif kind == 'm':
            base = self.value(body)
            g = base.groupby(self.spell)
            v = base - pd.concat([g.shift(k) for k in (1, 2, 3)], axis=1).mean(axis=1, skipna=True)
        else:
            a, b = body.split('|')
            v = sd(self.value(a), self.value(b))
        return v.replace([np.inf, -np.inf], np.nan)

    def ranks(self, specs):
        """Within-year percentile ranks, (rows, len(specs)) float32."""
        out = np.empty((len(self.years), len(specs)), dtype=np.float32)
        for i, s in enumerate(specs):
            out[:, i] = self.value(s).groupby(self.years).rank(pct=True).to_numpy(dtype=np.float32)
        return out


# ----------------------------------------------------------------------------- reduce: OpenFE-style
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logloss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class InnerFold:
    """One inner fold of the window: sampled training rows, base margins, validation rows."""

    def __init__(self, X, y_full, paaer, itr, iva, seed):
        rng = np.random.RandomState(seed)
        y_tr = mask_serial(y_full, paaer, itr, iva)
        pos, neg = itr[y_tr == 1], itr[y_tr == 0]
        n_neg = min(len(neg), RATIO * len(pos))
        self.rows = np.sort(np.concatenate([pos, rng.choice(neg, n_neg, replace=False)]))
        lab = dict(zip(itr, y_tr))
        self.y = np.array([lab[r] for r in self.rows])
        self.log_s = math.log(n_neg / len(neg))                  # sampled odds = true odds / s
        sub_neg = self.rows[self.y == 0]
        keep = rng.choice(sub_neg, int(round(RUNG1_NEG * len(sub_neg))), replace=False)
        self.rung1 = np.isin(self.rows, np.concatenate([self.rows[self.y == 1], keep]))
        self.log_r1 = math.log(len(keep) / len(sub_neg))
        base = lgbm_factory(TREE_LGBM, 1)(seed).fit(X[self.rows], self.y)
        self.m_tr = base.predict(X[self.rows], raw_score=True)
        self.va = iva
        self.y_va = y_full[iva]
        self.m_va = base.predict(X[iva], raw_score=True) + self.log_s
        self.ll_base = logloss(self.y_va, sigmoid(self.m_va))
        self.seed = seed

    def gain(self, col, rung):
        """Drop in validation log loss from boosting the base margin on one candidate column."""
        sel = self.rung1 if rung == 1 else np.ones(len(self.rows), dtype=bool)
        init = self.m_tr[sel] - (self.log_r1 if rung == 1 else 0.0)
        m = LGBMClassifier(**BOOST, n_jobs=1, verbose=-1, random_state=self.seed)
        m.fit(col[self.rows][sel, None], self.y[sel], init_score=init)
        inc = m.predict(col[self.va][:, None], raw_score=True)
        return self.ll_base - logloss(self.y_va, sigmoid(self.m_va + inc))


def score_specs(panel, specs, folds, rung, threads):
    scores = np.zeros(len(specs))
    with ThreadPoolExecutor(threads) as pool:
        for s in range(0, len(specs), CHUNK):
            block = panel.ranks(specs[s:s + CHUNK])
            jobs = [pool.submit(lambda j: np.mean([f.gain(block[:, j], rung) for f in folds]), j) for j in range(block.shape[1])]
            scores[s:s + CHUNK] = [j.result() for j in jobs]
    return scores


def halve(panel, specs, folds, threads):
    """Successive halving: all candidates on a quarter of the non-frauds, the top fifth on all rows."""
    s1 = score_specs(panel, specs, folds, 1, threads)
    top = [specs[i] for i in np.argsort(-s1, kind='stable')[:max(1, math.ceil(RUNG1_KEEP * len(specs)))]]
    s2 = score_specs(panel, top, folds, 2, threads)
    order = np.argsort(-s2, kind='stable')[:SURVIVORS]
    return [top[i] for i in order], {top[i]: float(s2[i]) for i in order}


def by_gain(panel, X, fold, specs, seed):
    """OpenFE stage 2: rank the survivors by split gain in a model that also has the existing features."""
    G = panel.ranks(specs)
    Z = np.c_[X[fold.rows], G[fold.rows]]
    m = lgbm_factory(TREE_LGBM, 4)(seed).fit(Z, fold.y)
    gain = m.booster_.feature_importance('gain')[X.shape[1]:]
    return [specs[i] for i in np.argsort(-gain, kind='stable')], {specs[i]: float(gain[i]) for i in range(len(specs))}


def select_task(key, family, seed, n_jobs):
    c = ctx()
    tr, te = split(key)
    y_full = c.y.copy()
    y_full[tr] = mask_serial(c.y, c.paaer, tr, te)
    D, feats = frame(tr)
    X = D[feats].values.astype(np.float32)
    L = c.years[tr].max()
    folds = [InnerFold(X, y_full, c.paaer, tr[c.years[tr] <= L - g - 1], tr[c.years[tr] == L - g + 1], seed) for g in (2, 1)]
    panel = Panel()
    r1, r1_score = halve(panel, round1_specs(), folds, n_jobs)
    if FAMILIES[family] == 1:
        order, gain = by_gain(panel, X, folds[-1], r1, seed)
        return {'order': order, 'gain': gain, 'round1': r1_score}
    first = load(select_path(key, 'G1', seed))['order'][:ROUND2_INPUTS]
    r2, r2_score = halve(panel, round2_specs(first), folds, n_jobs)
    order, gain = by_gain(panel, X, folds[-1], r1 + r2, seed)
    return {'order': order, 'gain': gain, 'round1': r1_score, 'round2': r2_score}


def select_path(key, family, seed):
    return path('fg', 'select', f'{family}_{key}_s{seed}.pkl')


# ----------------------------------------------------------------------------- members with the new features
def member_task(key, family, k, member, seed, n_jobs):
    c = ctx()
    tr, te = split(key)
    ytr = mask_serial(c.y, c.paaer, tr, te)
    D, feats = frame(tr)
    X = D[feats].values
    if k:
        X = np.c_[X, Panel().ranks(load(select_path(key, family, seed))['order'][:k])]
    if member == 'LR':
        return LogReg().fit(X[tr], ytr).predict_proba(X[te])
    p = TREE_LGBM if member == 'LGBM' else TREE_XGB
    make, sd_ = (lgbm_factory(p, n_jobs), seed) if member == 'LGBM' else (xgb_factory(p, n_jobs), seed + 1)
    return WeightedBagger(make, N_BAGS, p['ratio'], sd_).fit(X[tr], ytr, recency_weights(c.years[tr])).predict_proba(X[te])


def member_path(key, family, k, member, seed):
    return path('fg', 'member', f'{family}_k{k}_{key}_{member}_s{seed}.pkl')


def candidate(key, family, k, seed):
    from run_auc import tt_score
    return ensemble_scores([load(member_path(key, family, k, m, seed)) for m in MEMBERS] + [tt_score(key, seed)])


def run(stage_keys, settings, seeds):
    """Selection first (G2 needs G1), then the members."""
    fams = {f for f, _ in settings} | ({'G1'} if any(f == 'G2' for f, _ in settings) else set())
    for fam in ('G1', 'G2'):
        if fam in fams:
            run_tasks([(select_task, select_path(k, fam, s), dict(key=k, family=fam, seed=s)) for k in stage_keys for s in seeds],
                      n_workers=3, threads=5)
    run_tasks([(member_task, member_path(k, f, kk, m, s), dict(key=k, family=f, k=kk, member=m, seed=s))
               for k in stage_keys for s in seeds for f, kk in settings for m in MEMBERS], n_workers=4, threads=4)


def settings_all():
    return [(f, k) for f in FAMILIES for k in KS]


def stage_screen():
    run_tasks(reference_tasks(keys('cv'), (SCREEN_SEED,)), n_workers=3, threads=5)
    run(keys('cv'), settings_all(), (SCREEN_SEED,))
    ref = cv_report({SCREEN_SEED: {k: reference(k, SCREEN_SEED) for k in keys('cv')}})
    rows = [{'family': f, 'k': k, **cv_report({SCREEN_SEED: {kk: candidate(kk, f, k, SCREEN_SEED) for kk in keys('cv')}})}
            for f, k in settings_all()]
    for r in rows:
        r['gain'] = r['cv_auc'] - ref['cv_auc']
    json.dump({'reference': ref, 'candidates': rows}, open(os.path.join(OUT, 'cv_fg_screen.json'), 'w'), indent=2)
    print('reference', round(ref['cv_auc'], 4))
    for r in rows:
        print(f"{r['family']} k={r['k']:>2}  CV AUC {r['cv_auc']:.4f}  gain {r['gain']:+.4f}  folds {r['fold_auc']}", flush=True)


def stage_check():
    screen = json.load(open(os.path.join(OUT, 'cv_fg_screen.json')))['candidates']
    best = [max((r for r in screen if r['family'] == f), key=lambda r: r['gain']) for f in FAMILIES]
    settings = [(r['family'], r['k']) for r in best]
    run(keys('cv'), settings, CHECK_SEEDS)
    ref = cv_report({s: {k: reference(k, s) for k in keys('cv')} for s in CHECK_SEEDS})
    rows = []
    for (f, k), b in zip(settings, best):
        r = cv_report({s: {kk: candidate(kk, f, k, s) for kk in keys('cv')} for s in CHECK_SEEDS})
        rows.append({'family': f, 'k': k, 'screen_gain': b['gain'], 'gain': r['cv_auc'] - ref['cv_auc'],
                     'gate': bool(r['cv_auc'] - ref['cv_auc'] >= GATE), **r})
    passed = sorted((r for r in rows if r['gate']), key=lambda r: -r['gain'])[:2]
    out = {'reference': ref, 'candidates': rows, 'advance': [[r['family'], r['k']] for r in passed]}
    json.dump(out, open(os.path.join(OUT, 'cv_fg.json'), 'w'), indent=2)
    print('reference', round(ref['cv_auc'], 4))
    for r in rows:
        print(f"{r['family']} k={r['k']:>2}  CV AUC {r['cv_auc']:.4f}  gain {r['gain']:+.4f}  gate {r['gate']}", flush=True)
    print('advance:', out['advance'], flush=True)


def stage_test():
    adv = [tuple(a) for a in json.load(open(os.path.join(OUT, 'cv_fg.json')))['advance']]
    assert adv, 'nothing passed the validation gate: the test years are not used (PROTOCOL_FG.md)'
    run(keys('test'), adv, SEEDS)


def stage_compare():
    from compare import MIN_FRAUDS, N_BOOT, PERIODS, holm, stacked
    from evaluate import BOOT_METRICS, bootstrap_period, per_year_metrics, period_mean
    from common import TEST_YEARS
    c = ctx()
    adv = [tuple(a) for a in json.load(open(os.path.join(OUT, 'cv_fg.json')))['advance']]
    tk = keys('test')
    te = np.concatenate([split(k)[1] for k in tk])
    y, yrs = c.y[te], c.years[te]
    pos_key = np.where(np.isnan(c.paaer), -c.gvkey, c.paaer)[te]
    names = [f'{f}_k{k}' for f, k in adv]
    preds = {('reference', s): stacked({k: reference(k, s) for k in tk}, tk) for s in SEEDS}
    for (f, k), n in zip(adv, names):
        for s in SEEDS:
            preds[(n, s)] = stacked({kk: candidate(kk, f, k, s) for kk in tk}, tk)
    yearly = {key: per_year_metrics(y, p, yrs) for key, p in preds.items()}
    use = [t for t in TEST_YEARS if y[yrs == t].sum() >= MIN_FRAUDS]
    boot = bootstrap_period(y, preds, yrs, pos_key, c.gvkey[te], use, n_boot=N_BOOT, seed=42)

    def mean(n, s, lo, hi):
        return period_mean(yearly[(n, s)], lo, hi, MIN_FRAUDS)

    tests = []
    for n in names:
        d = np.mean([boot[(n, s)] - boot[('reference', s)] for s in SEEDS], axis=0)
        for j, met in enumerate(BOOT_METRICS):
            seed_gain = [mean(n, s, 2003, 2012)[met] - mean('reference', s, 2003, 2012)[met] for s in SEEDS]
            per = {lbl: float(np.mean([mean(n, s, lo, hi)[met] - mean('reference', s, lo, hi)[met] for s in SEEDS])) for lbl, (lo, hi) in PERIODS.items()}
            tests.append({'candidate': n, 'metric': met, 'gain': per['2003-2012'], 'ci_lo': float(np.percentile(d[:, j], 2.5)),
                          'ci_hi': float(np.percentile(d[:, j], 97.5)), 'p': float(((d[:, j] <= 0).sum() + 1) / (N_BOOT + 1)),
                          'gain_2003_2008': per['2003-2008'], 'gain_2009_2012': per['2009-2012'],
                          'min_seed_gain': float(min(seed_gain)), 'max_seed_gain': float(max(seed_gain)),
                          'reference_value': float(np.mean([mean('reference', s, 2003, 2012)[met] for s in SEEDS]))})
    t = pd.DataFrame(tests)
    t['p_holm'] = holm(t['p'].values)
    t.to_csv(os.path.join(OUT, 'fg_test_comparison.csv'), index=False)
    print(t.round(4).to_string(index=False), flush=True)
    verdicts = {}
    for n, g in t.groupby('candidate'):
        a = g[g.metric == 'AUC'].iloc[0]
        harm = bool((g.p >= 0.90).any())
        verdicts[n] = {'adopt': bool(a.p <= 0.05 and not harm and a.min_seed_gain > 0 and a.gain_2003_2008 > 0 and a.gain_2009_2012 > 0),
                       'auc': float(a.reference_value + a.gain), 'auc_gain': float(a.gain), 'p': float(a.p), 'p_holm': float(a.p_holm), 'harm': harm}
    json.dump(verdicts, open(os.path.join(OUT, 'fg_test_verdicts.json'), 'w'), indent=2)
    print(json.dumps(verdicts, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['screen', 'check', 'test', 'compare'], required=True)
    a = ap.parse_args()
    ctx()
    market()
    {'screen': stage_screen, 'check': stage_check, 'test': stage_test, 'compare': stage_compare}[a.stage]()


if __name__ == '__main__':
    main()

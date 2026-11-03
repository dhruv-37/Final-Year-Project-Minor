"""Stage 2 of PROTOCOL.md: every stored candidate against the reference, with the adoption rule applied.

    python experiments/compare.py
"""
import glob
import json

import numpy as np
import pandas as pd

from common import CACHE, OUT, SEEDS, TEST_YEARS, ctx, keys, load, os, split
from evaluate import BOOT_METRICS, bootstrap_period, per_year_metrics, period_mean
from scoring import reference_ensemble

N_BOOT = 1000
MIN_FRAUDS = 15
CV_GATE = 0.002
PERIODS = {'2003-2012': (2003, 2012), '2003-2008': (2003, 2008), '2009-2012': (2009, 2012)}


def stacked(by_key, rows_of):
    return np.concatenate([by_key[k] for k in rows_of])


def holm(pvals):
    order = np.argsort(pvals)
    adj, top = np.zeros(len(pvals)), 0.0
    for r, j in enumerate(order):
        top = max(top, min(1.0, (len(pvals) - r) * pvals[j]))
        adj[j] = top
    return adj


def cv_gains():
    """Stage 1: each candidate's CV AUC gain over the reference, taken from the runners' reports."""
    def report(name):
        p = os.path.join(OUT, f'cv_{name}.json')
        return json.load(open(p)) if os.path.exists(p) else {}
    gains = {}
    drfe, pu, deep = report('drfe'), report('pu'), report('deep')
    for r in drfe.get('candidates', []):
        gains[f"DRFE_tau{r['tau']}_{r['rule']}"] = r['cv_auc'] - drfe['reference']['cv_auc']
    for r in pu.get('candidates', []):
        gains[f"PU_pi{r['pi']}_g{r['gamma']}_{r['mode']}"] = r['cv_auc'] - pu['reference']['cv_auc']
    for a, r in deep.items():
        if a != 'reference':
            gains[f'{a}_alone'] = r['alone']['cv_auc'] - deep['reference']['cv_auc']
            gains[f'reference+{a}'] = r['with_reference']['cv_auc'] - deep['reference']['cv_auc']
    return gains


def main():
    c = ctx()
    test_keys = keys('test')
    te = np.concatenate([split(k)[1] for k in test_keys])
    y, yrs = c.y[te], c.years[te]
    pos_key = np.where(np.isnan(c.paaer), -c.gvkey, c.paaer)[te]
    names = sorted({os.path.basename(f).rsplit('_s', 1)[0] for f in glob.glob(os.path.join(CACHE, 'candidates', '*.pkl'))})
    preds = {('reference', s): stacked({k: reference_ensemble(k, s) for k in test_keys}, test_keys) for s in SEEDS}
    for n in names:
        for s in SEEDS:
            preds[(n, s)] = stacked(load(os.path.join(CACHE, 'candidates', f'{n}_s{s}.pkl')), test_keys)
    yearly = {key: per_year_metrics(y, p, yrs) for key, p in preds.items()}
    use = [t for t in TEST_YEARS if y[yrs == t].sum() >= MIN_FRAUDS]
    boot = bootstrap_period(y, preds, yrs, pos_key, c.gvkey[te], use, n_boot=N_BOOT, seed=42)

    def mean(n, s, lo, hi):
        return period_mean(yearly[(n, s)], lo, hi, MIN_FRAUDS)

    rows = []
    for n in ['reference'] + names:
        for lbl, (lo, hi) in PERIODS.items():
            m = pd.concat([mean(n, s, lo, hi) for s in SEEDS], axis=1).mean(axis=1)
            rows.append({'candidate': n, 'period': lbl, **m.round(4).to_dict()})
    means = pd.DataFrame(rows)
    means.to_csv(os.path.join(OUT, 'test_means.csv'), index=False)
    print(means[means.period == '2003-2012'].to_string(index=False), flush=True)

    tests = []
    for n in names:
        d = np.mean([boot[(n, s)] - boot[('reference', s)] for s in SEEDS], axis=0)
        for j, met in enumerate(BOOT_METRICS):
            seed_gain = [mean(n, s, 2003, 2012)[met] - mean('reference', s, 2003, 2012)[met] for s in SEEDS]
            per = {lbl: float(np.mean([mean(n, s, lo, hi)[met] - mean('reference', s, lo, hi)[met] for s in SEEDS]))
                   for lbl, (lo, hi) in PERIODS.items()}
            tests.append({'candidate': n, 'metric': met, 'gain': per['2003-2012'],
                          'ci_lo': float(np.percentile(d[:, j], 2.5)), 'ci_hi': float(np.percentile(d[:, j], 97.5)),
                          'p': float(((d[:, j] <= 0).sum() + 1) / (N_BOOT + 1)), 'gain_2003_2008': per['2003-2008'],
                          'gain_2009_2012': per['2009-2012'], 'min_seed_gain': float(min(seed_gain)),
                          'max_seed_gain': float(max(seed_gain))})
    t = pd.DataFrame(tests)
    verdicts = {}
    gains = cv_gains()
    if len(t):
        t['p_holm'] = holm(t['p'].values)
        for n, g in t.groupby('candidate'):
            passing = g[(g.p <= 0.05) & (g.min_seed_gain > 0) & (g.gain_2003_2008 > 0) & (g.gain_2009_2012 > 0)]
            harm = bool((g.p >= 0.90).any())
            gate = gains[n] >= CV_GATE  # a test-year result cannot rescue a candidate that failed validation
            adopt = gate and len(passing) > 0 and not harm
            verdicts[n] = {'adopt': adopt, 'cv_gain': round(gains[n], 4), 'gate': gate, 'harm': harm,
                           'passing_metrics': list(passing.metric),
                           'survives_holm': bool((passing.p_holm <= 0.05).any()) if adopt else False}
        t.to_csv(os.path.join(OUT, 'test_comparison.csv'), index=False)
        print(t.round(4).to_string(index=False), flush=True)
    json.dump(verdicts, open(os.path.join(OUT, 'test_verdicts.json'), 'w'), indent=2)
    print(json.dumps(verdicts, indent=2))


if __name__ == '__main__':
    main()

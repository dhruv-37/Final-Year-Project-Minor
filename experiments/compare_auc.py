"""Stage 2 of PROTOCOL_AUC.md: candidates with test scores in the cache against the previous ensemble.
The recorded Holm p-values cover 3 candidates; rerun with fewer, they come out smaller.

    python experiments/compare_auc.py
"""
import glob
import json

import numpy as np
import pandas as pd

from common import CACHE, OUT, SEEDS, TEST_YEARS, ctx, keys, load, os, split
from compare import MIN_FRAUDS, N_BOOT, PERIODS, holm, stacked
from evaluate import BOOT_METRICS, bootstrap_period, per_year_metrics, period_mean
from run_auc import reference

FOLDER = os.path.join(CACHE, 'auc', 'candidates')


def main():
    c = ctx()
    test_keys = keys('test')
    te = np.concatenate([split(k)[1] for k in test_keys])
    y, yrs = c.y[te], c.years[te]
    pos_key = np.where(np.isnan(c.paaer), -c.gvkey, c.paaer)[te]
    preds = {('reference', s): stacked({k: reference(k, s) for k in test_keys}, test_keys) for s in SEEDS}
    names = {os.path.basename(f).rsplit('_s', 1)[0] for f in glob.glob(os.path.join(FOLDER, '*.pkl'))}
    for n in names:
        for s in SEEDS:
            preds[(n, s)] = stacked(load(os.path.join(FOLDER, f'{n}_s{s}.pkl')), test_keys)
    yearly = {key: per_year_metrics(y, p, yrs) for key, p in preds.items()}
    use = [t for t in TEST_YEARS if y[yrs == t].sum() >= MIN_FRAUDS]
    boot = bootstrap_period(y, preds, yrs, pos_key, c.gvkey[te], use, n_boot=N_BOOT, seed=42)

    def mean(n, s, lo, hi):
        return period_mean(yearly[(n, s)], lo, hi, MIN_FRAUDS)

    rows = []
    for n in ['reference'] + sorted(names):
        for lbl, (lo, hi) in PERIODS.items():
            m = pd.concat([mean(n, s, lo, hi) for s in SEEDS], axis=1).mean(axis=1)
            rows.append({'candidate': n, 'period': lbl, **m.round(4).to_dict()})
    means = pd.DataFrame(rows)
    means.to_csv(os.path.join(OUT, 'auc_test_means.csv'), index=False)
    print(means[means.period == '2003-2012'].to_string(index=False), flush=True)
    per_year = pd.concat({n: pd.concat([yearly[(n, s)].set_index('year')['AUC'] for s in SEEDS], axis=1).mean(axis=1) for n in ['reference'] + sorted(names)}, axis=1)
    per_year.round(4).to_csv(os.path.join(OUT, 'auc_test_by_year.csv'))
    print(per_year.round(4).to_string(), flush=True)

    tests = []
    for n in sorted(names):
        d = np.mean([boot[(n, s)] - boot[('reference', s)] for s in SEEDS], axis=0)
        for j, met in enumerate(BOOT_METRICS):
            seed_gain = [mean(n, s, 2003, 2012)[met] - mean('reference', s, 2003, 2012)[met] for s in SEEDS]
            per = {lbl: float(np.mean([mean(n, s, lo, hi)[met] - mean('reference', s, lo, hi)[met] for s in SEEDS]))
                   for lbl, (lo, hi) in PERIODS.items()}
            tests.append({'candidate': n, 'metric': met, 'gain': per['2003-2012'], 'ci_lo': float(np.percentile(d[:, j], 2.5)),
                          'ci_hi': float(np.percentile(d[:, j], 97.5)), 'p': float(((d[:, j] <= 0).sum() + 1) / (N_BOOT + 1)),
                          'gain_2003_2008': per['2003-2008'], 'gain_2009_2012': per['2009-2012'],
                          'min_seed_gain': float(min(seed_gain)), 'max_seed_gain': float(max(seed_gain))})
    t = pd.DataFrame(tests)
    t['p_holm'] = holm(t['p'].values)
    t.to_csv(os.path.join(OUT, 'auc_test_comparison.csv'), index=False)
    print(t.round(4).to_string(index=False), flush=True)
    verdicts = {}
    for n, g in t.groupby('candidate'):
        a = g[g.metric == 'AUC'].iloc[0]
        harm = bool((g.p >= 0.90).any())
        adopt = bool(a.p <= 0.05 and not harm and a.min_seed_gain > 0 and a.gain_2003_2008 > 0 and a.gain_2009_2012 > 0)
        verdicts[n] = {'adopt': adopt, 'auc_gain': float(a.gain), 'p': float(a.p), 'p_holm': float(a.p_holm), 'harm': harm}
    json.dump(verdicts, open(os.path.join(OUT, 'auc_test_verdicts.json'), 'w'), indent=2)
    print(json.dumps(verdicts, indent=2))


if __name__ == '__main__':
    main()

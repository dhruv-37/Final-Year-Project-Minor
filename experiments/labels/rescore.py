"""PROTOCOL_LABELS.md step 1: score the saved test-year predictions against the completed labels.

No model is refit. results/test_predictions.csv (final run, seed 42) is scored twice, against the paper's
labels and against experiments/labels/labels_completed.csv.

    python experiments/labels/rescore.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from evaluate import bootstrap_period, per_year_metrics  # noqa: E402

BASE, PROP = 'Baseline: RUSBoost (paper, raw 28)', 'PROPOSED: ensemble'
OTHER = ['Ensemble without firm history', 'Previous ensemble (no market features or recency weights)']


def main():
    p = pd.read_csv(os.path.join(ROOT, 'results', 'test_predictions.csv'))
    lab = pd.read_csv(os.path.join(HERE, 'labels_completed.csv')).rename(columns={'misstate': 'y_new', 'p_aaer': 'case_new'})
    orig = pd.read_csv(os.path.join(ROOT, 'data', 'data_FraudDetection_JAR2020.csv'), usecols=['gvkey', 'fyear', 'p_aaer'])
    p = p.merge(lab, on=['gvkey', 'fyear'], how='left').merge(orig, on=['gvkey', 'fyear'], how='left')
    assert p['y_new'].notna().all()
    print(f'test rows {len(p)}; labels changed: {int((p.y_new != p.misstate).sum())} (all 0 -> 1: {bool((p.y_new >= p.misstate).all())})')
    res = {}
    models = [BASE, PROP] + OTHER
    for name, ycol, ccol in (('paper labels', 'misstate', 'p_aaer'), ('completed labels', 'y_new', 'case_new')):
        y, yrs = p[ycol].values.astype(int), p['fyear'].values
        yearly = {m: per_year_metrics(y, p[m].values, yrs).set_index('year') for m in models}
        fr = yearly[PROP]['frauds']
        r = {'frauds_by_year': {int(k): int(v) for k, v in fr.items()},
             'AUC_by_year': {m: {int(k): round(float(v), 4) for k, v in yearly[m]['AUC'].items()} for m in (BASE, PROP)}}
        for lo, hi in ((2003, 2012), (2003, 2014)):
            use = [int(t) for t in range(lo, hi + 1) if fr.get(t, 0) >= 15]
            means = {m: yearly[m].loc[use, ['AUC', 'NDCG@5%', 'Sens@5%', 'Prec@1%']].mean().round(4).to_dict() for m in models}
            key = np.where(p[ccol].isna(), -p['gvkey'], p[ccol]).astype(float)
            boot = bootstrap_period(y, {m: p[m].values for m in (BASE, PROP)}, yrs, key, p['gvkey'].values, use, n_boot=1000, seed=0)
            d = boot[PROP][:, 0] - boot[BASE][:, 0]
            r[f'{lo}-{hi}'] = {'years_used': use, 'means': means,
                               'AUC_gain': round(means[PROP]['AUC'] - means[BASE]['AUC'], 4),
                               'AUC_gain_CI95': [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                               'P(gain<=0)': round(float((d <= 0).mean()), 4)}
        res[name] = r
        print(f'\n=== {name} ===')
        print('frauds by year:', r['frauds_by_year'])
        print('AUC by year, proposed:', r['AUC_by_year'][PROP])
        print('AUC by year, baseline:', r['AUC_by_year'][BASE])
        for per in ('2003-2012', '2003-2014'):
            q = r[per]
            print(f'{per} over {q["years_used"]}:')
            for m in models:
                print(f'   {m[:45]:45s} AUC {q["means"][m]["AUC"]:.4f}  NDCG@5% {q["means"][m]["NDCG@5%"]:.4f}  '
                      f'Sens@5% {q["means"][m]["Sens@5%"]:.4f}  Prec@1% {q["means"][m]["Prec@1%"]:.4f}')
            print(f'   gain over baseline {q["AUC_gain"]:+.4f}, 95% CI {q["AUC_gain_CI95"]}, P(gain<=0) {q["P(gain<=0)"]}')
    json.dump(res, open(os.path.join(ROOT, 'results', 'experiments', 'labels_rescore.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()

"""PROTOCOL_LABELS.md step 2: compare the model retrained on the completed labels with the re-scored one.

Run after
    python run_pipeline.py --data data/data_completed_labels.csv --reuse results --out results_labels --skip_final
which retrains every model on the completed labels (tuning reused: the labels change only fiscal 2011-2014,
after the tuning folds). Both prediction files are scored against the completed labels.

    python experiments/labels/compare_retrained.py
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
COLS = ['AUC', 'NDCG@5%', 'Sens@5%', 'Prec@1%']


def main():
    lab = pd.read_csv(os.path.join(HERE, 'labels_completed.csv')).rename(columns={'misstate': 'y', 'p_aaer': 'case'})
    res = {}
    for name, path in (('re-scored (trained on paper labels)', 'results'), ('retrained on completed labels', 'results_labels')):
        p = pd.read_csv(os.path.join(ROOT, path, 'test_predictions.csv')).drop(columns=['misstate'], errors='ignore')
        p = p.merge(lab, on=['gvkey', 'fyear'], how='left')
        assert p['y'].notna().all()
        y, yrs = p['y'].values.astype(int), p['fyear'].values
        yearly = {m: per_year_metrics(y, p[m].values, yrs).set_index('year') for m in (BASE, PROP)}
        r = {'AUC_by_year': {m: {int(k): round(float(v), 4) for k, v in yearly[m]['AUC'].items()} for m in (BASE, PROP)}}
        key = np.where(p['case'].isna(), -p['gvkey'], p['case']).astype(float)
        for lo, hi in ((2003, 2012), (2003, 2014)):
            use = list(range(lo, hi + 1))
            means = {m: yearly[m].loc[use, COLS].mean().round(4).to_dict() for m in (BASE, PROP)}
            boot = bootstrap_period(y, {m: p[m].values for m in (BASE, PROP)}, yrs, key, p['gvkey'].values, use,
                                    n_boot=1000, seed=0)
            d = boot[PROP][:, 0] - boot[BASE][:, 0]
            r[f'{lo}-{hi}'] = {'means': means, 'AUC_gain': round(means[PROP]['AUC'] - means[BASE]['AUC'], 4),
                               'AUC_gain_CI95': [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                               'P(gain<=0)': round(float((d <= 0).mean()), 4)}
        res[name] = r
        print(f'\n=== {name} ===')
        print('AUC by year 2009-2014, proposed:', {k: v for k, v in r['AUC_by_year'][PROP].items() if k >= 2009})
        print('AUC by year 2009-2014, baseline:', {k: v for k, v in r['AUC_by_year'][BASE].items() if k >= 2009})
        for per in ('2003-2012', '2003-2014'):
            q = r[per]
            for m in (BASE, PROP):
                print(f'{per} {m[:35]:35s} ' + '  '.join(f'{c} {q["means"][m][c]:.4f}' for c in COLS))
            print(f'{per} gain {q["AUC_gain"]:+.4f}, 95% CI {q["AUC_gain_CI95"]}, P(gain<=0) {q["P(gain<=0)"]}')
    json.dump(res, open(os.path.join(ROOT, 'results', 'experiments', 'labels_retrained.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()

"""PROTOCOL_CASES.md: scores of the final run split by when each test fraud's AAER case started.

    python experiments/cases/case_split.py
Writes results/experiments/case_split.json and case_split.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from evaluate import _clusters, _draw  # noqa: E402

BASE, PROP = 'Baseline: RUSBoost (paper, raw 28)', 'PROPOSED: ensemble'
NOTT, PREV, TT = 'Ensemble without firm history', 'Previous ensemble (no market features or recency weights)', 'Temporal Transformer (firm history)'
MODELS = [BASE, PROP, NOTT, PREV, TT]
SHORT = {BASE: 'baseline', PROP: 'proposed', NOTT: 'no_firm_history', PREV: 'previous_ensemble', TT: 'TT_alone'}
GAINS = {'proposed_vs_baseline': (PROP, BASE), 'firm_history_gain': (PROP, NOTT), 'market_recency_gain': (PROP, PREV)}
GROUPS = ['continuing', 'gap-start', 'new', 'all']
N_BOOT = 1000


def load(source):
    p = pd.read_csv(os.path.join(ROOT, 'results', 'test_predictions.csv'))
    p = p[p.fyear.between(2003, 2012)].reset_index(drop=True)
    d = pd.read_csv(os.path.join(ROOT, 'data', 'data_FraudDetection_JAR2020.csv'), usecols=['gvkey', 'fyear', 'misstate', 'p_aaer'])
    p = p.merge(d[['gvkey', 'fyear', 'p_aaer']], on=['gvkey', 'fyear'], how='left')
    start = d[d.misstate == 1].groupby('p_aaer').fyear.min()
    if source == 'aaer file + data':
        a = pd.read_csv(os.path.join(ROOT, 'data', 'external', 'fresh', 'jarfraud', 'AAER_firm_year.csv'))
        start = pd.concat([start, a.groupby('P_AAER').YEARA.min()], axis=1).min(axis=1)
    lag = p.fyear - p.p_aaer.map(start)
    p['group'] = np.where(p.misstate != 1, '', np.where(lag >= 2, 'continuing', np.where(lag == 1, 'gap-start', 'new')))
    assert (lag[p.misstate == 1] >= 0).all()
    return p


def stats(y, s, top5, top1):
    """Pairs of (fraud, non-fraud) ranked correctly, frauds in the top 5% / 1%, frauds: sums for pooling."""
    n1 = y.sum()
    r = rankdata(s)
    return np.array([r[y == 1].sum() - n1 * (n1 + 1) / 2, n1 * (len(y) - n1), top5[y == 1].sum(), top1[y == 1].sum(), n1])


def metrics(p, rows_by_year):
    """Pooled AUC / Sens@5% / Sens@1% per group and model, rows_by_year = {year: (fraud rows of group, nonfraud rows)}."""
    out = {}
    for m in MODELS:
        tot = np.zeros(5)
        for yr, (pos, neg, s_all) in rows_by_year.items():
            if not len(pos):
                continue
            s = p[m].values
            sc = s_all[m]
            idx = np.concatenate([pos, neg])
            y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            tot += stats(y, s[idx], s[idx] >= sc[0], s[idx] >= sc[1])
        out[m] = np.array([tot[0] / tot[1], tot[2] / tot[4], tot[3] / tot[4]])
    return out


def run(source):
    p = load(source)
    cut = {yr: {m: (np.quantile(g[m], 0.95), np.quantile(g[m], 0.99)) for m in MODELS} for yr, g in p.groupby('fyear')}
    key = p.p_aaer.fillna(-p.gvkey).values
    res = {'counts': p[p.misstate == 1].groupby(['fyear', 'group']).size().unstack(fill_value=0).to_dict(orient='index')}
    rng = np.random.RandomState(0)
    for g in GROUPS:
        base_rows, clus = {}, {}
        for yr, d in p.groupby('fyear'):
            pos = d.index[(d.misstate == 1) & ((d.group == g) if g != 'all' else True)].values
            neg = d.index[d.misstate == 0].values
            base_rows[yr] = (pos, neg, cut[yr])
            clus[yr] = (pos, _clusters(key[pos]) if len(pos) else None, neg, _clusters(p.gvkey.values[neg]))
        point = metrics(p, base_rows)
        boot = []
        for b in range(N_BOOT):
            rows = {}
            for yr, (pos, cp, neg, cn) in clus.items():
                rows[yr] = (_draw(rng, pos, cp) if len(pos) else pos, _draw(rng, neg, cn), cut[yr])
            boot.append(metrics(p, rows))
        r = {'frauds': int(sum(len(v[0]) for v in base_rows.values())),
             'metrics': {SHORT[m]: dict(zip(['AUC', 'Sens@5%', 'Sens@1%'], np.round(point[m], 4).tolist())) for m in MODELS}}
        for name, (a, bm) in GAINS.items():
            d = np.array([x[a] - x[bm] for x in boot])
            r[name] = {met: {'gain': round(float(point[a][j] - point[bm][j]), 4),
                             'CI95': [round(float(np.percentile(d[:, j], 2.5)), 4), round(float(np.percentile(d[:, j], 97.5)), 4)]}
                       for j, met in enumerate(['AUC', 'Sens@5%', 'Sens@1%'])}
        res[g] = r
    return res


def main():
    out, rows = {}, []
    for source in ('aaer file + data', 'data only'):
        r = out[source] = run(source)
        print(f'\n=== case start from {source} ===')
        for g in GROUPS:
            q = r[g]
            print(f'{g:11s} frauds {q["frauds"]:4d}  ' + '  '.join(f'{k} AUC {v["AUC"]:.3f} S5 {v["Sens@5%"]:.3f} S1 {v["Sens@1%"]:.3f}'
                                                                for k, v in q['metrics'].items() if k in ('baseline', 'proposed', 'TT_alone')))
            for name in GAINS:
                a = q[name]['AUC']
                print(f'            {name:22s} AUC {a["gain"]:+.4f} {a["CI95"]}   Sens@5% {q[name]["Sens@5%"]["gain"]:+.4f} {q[name]["Sens@5%"]["CI95"]}')
            for k, v in q['metrics'].items():
                rows.append({'case_start': source, 'group': g, 'frauds': q['frauds'], 'model': k, **v})
    json.dump(out, open(os.path.join(ROOT, 'results', 'experiments', 'case_split.json'), 'w'), indent=2, default=str)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'results', 'experiments', 'case_split.csv'), index=False)


if __name__ == '__main__':
    main()

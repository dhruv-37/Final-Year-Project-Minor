"""Steps 1 and 2 of PROTOCOL_FRESH.md: score the saved model on the XBRL-built firm-years.

    python experiments/fresh/score_fresh.py bridge    # fiscal 2012-2014, Compustat inputs vs XBRL inputs
    python experiments/fresh/score_fresh.py fresh     # fiscal 2015-2018 (+2019-2020), run once

The model is results/final_models.joblib, loaded and used as it is. Nothing is fitted here.
"""
import io
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, HERE)
from bridge_check import linked  # noqa: E402
from evaluate import bootstrap_period, per_year_metrics, rank_norm  # noqa: E402
from features import engineer, market_features  # noqa: E402
from models import ensemble_scores  # noqa: E402

FRESH = os.path.join(ROOT, 'data', 'external', 'fresh')
OUT = os.path.join(ROOT, 'results', 'experiments')
PRIMARY, LATE = (2015, 2016, 2017, 2018), (2019, 2020)
MIN_FRAUDS, N_BOOT = 15, 1000


def score(frame, art, years):
    """Risk scores for the given fiscal years. `frame` is in the paper's layout and holds the whole
    market, so within-year ranks and the final rank average span the same population as in the pipeline."""
    hist, _ = engineer(frame)
    gv, fy = hist['gvkey'].values, hist['fyear'].values
    raw = hist[art['winsor_cols']].astype(np.float32)
    D = pd.concat([art['winsorizer'].transform(raw), market_features(raw, gv, fy)], axis=1)
    X = D[art['features']].values
    parts = []
    for y in years:
        rows = np.where(fy == y)[0]
        s = {m: art['models'][m].predict_proba(X[rows]) for m in art['ensemble_members']}
        s['TT'] = art['temporal'].predict_scores(D, gv, fy, rows)
        part = hist.iloc[rows][['gvkey', 'fyear', 'at', 'prcc_f', 'csho']].reset_index(drop=True)
        for m, v in s.items():
            part[m] = rank_norm(v)
        part['ensemble'] = ensemble_scores(list(s.values()))
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def fresh_labels():
    path = os.path.join(HERE, 'aaer_labels_fresh.csv')
    lab = pd.read_csv(io.StringIO(''.join(ln for ln in open(path, encoding='utf-8') if not ln.startswith('#'))))
    return {(int(r.cik), int(y)): r.tier for r in lab.itertuples() for y in str(r.fyears).split(';')}


QUARTER_ONLY = {  # annual statements not misstated; used only in sensitivity check 5
    'DS Healthcare Group': (1463959, [2015]), 'Bausch Health (Valeant)': (885590, [2015]),
    'Osiris Therapeutics': (1360886, [2015]), 'Belden': (913142, [2017]), 'Mattel': (63276, [2017]),
    'Sequential Brands Group': (1648428, [2017]), 'Rollins': (84839, [2016, 2017]), 'Amyris': (1365916, [2018]),
    'Healthcare Services Group': (731012, [2015]), 'Mylan': (1623613, [2016]),
}


def pooled_auc_ci(y, p, firm, n_boot=N_BOOT, seed=0):
    """Pooled AUC with a bootstrap over firms (all of a firm's years move together)."""
    rng = np.random.RandomState(seed)
    ids, inv = np.unique(firm, return_inverse=True)
    rows = [np.where(inv == k)[0] for k in range(len(ids))]
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[k] for k in rng.randint(len(ids), size=len(ids))])
        if 0 < y[idx].sum() < len(idx):
            stats.append(roc_auc_score(y[idx], p[idx]))
    return float(roc_auc_score(y, p)), [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def summarise(s, y, years, col='ensemble'):
    """Yearly metrics, their mean over years with enough frauds, the pooled AUC, and both CIs."""
    y = np.asarray(y)
    m = s['fyear'].isin(years).values
    s, y = s[m].reset_index(drop=True), y[m]
    pym = per_year_metrics(y, s[col].values, s['fyear'].values)
    use = [int(t) for t in pym.loc[pym['frauds'] >= MIN_FRAUDS, 'year']]
    out = {'yearly': pym.round(4).to_dict('records'), 'years_used': use, 'frauds': int(y.sum()), 'rows': int(len(y))}
    if use:
        mean = pym[pym['year'].isin(use)][['AUC', 'NDCG@5%', 'Sens@5%']].mean()
        firm = s['gvkey'].values
        boot = bootstrap_period(y, {col: s[col].values}, s['fyear'].values, firm, firm, use, n_boot=N_BOOT, seed=0)[col]
        out['mean_yearly'] = {k: float(v) for k, v in mean.items()}
        out['mean_yearly_AUC_CI95'] = [float(np.percentile(boot[:, 0], 2.5)), float(np.percentile(boot[:, 0], 97.5))]
    if 0 < y.sum() < len(y):
        out['pooled_AUC'], out['pooled_AUC_CI95'] = pooled_auc_ci(y, s[col].values, s['gvkey'].values)
    return out


def show(title, r):
    line = f'{title}: rows {r["rows"]}, frauds {r["frauds"]}'
    if 'mean_yearly' in r:
        lo, hi = r['mean_yearly_AUC_CI95']
        line += (f' | mean yearly AUC {r["mean_yearly"]["AUC"]:.3f} [{lo:.3f}, {hi:.3f}] over {r["years_used"]}'
                 f' NDCG@5% {r["mean_yearly"]["NDCG@5%"]:.3f} Sens@5% {r["mean_yearly"]["Sens@5%"]:.3f}')
    if 'pooled_AUC' in r:
        line += f' | pooled AUC {r["pooled_AUC"]:.3f} [{r["pooled_AUC_CI95"][0]:.3f}, {r["pooled_AUC_CI95"][1]:.3f}]'
    print(line, flush=True)


def bridge(art):
    years = (2012, 2013, 2014)
    comp = pd.read_csv(os.path.join(ROOT, 'data', 'data_FraudDetection_JAR2020.csv'))
    xb_all = pd.read_csv(os.path.join(FRESH, 'xbrl_firm_years.csv'))
    link = linked()[['cik', 'fyear', 'gvkey']].rename(columns={'gvkey': 'gv'})
    s_c = score(comp, art, years)
    s_x = score(xb_all, art, years).rename(columns={'gvkey': 'cik'}).merge(link, on=['cik', 'fyear'])
    truth = comp[['gvkey', 'fyear', 'misstate']]
    both = s_c.merge(truth, on=['gvkey', 'fyear']).merge(
        s_x[['gv', 'fyear', 'ensemble']].rename(columns={'gv': 'gvkey', 'ensemble': 'ens_x'}), on=['gvkey', 'fyear'])
    full = s_c.merge(truth, on=['gvkey', 'fyear'])
    res = {'compustat_full_market': summarise(full, full['misstate'], years),
           'common_rows_compustat_inputs': summarise(both, both['misstate'], years),
           'common_rows_xbrl_inputs': summarise(both.assign(ensemble=both['ens_x']), both['misstate'], years),
           'rank_correlation_of_scores': float(both['ensemble'].corr(both['ens_x'], method='spearman'))}
    for k in list(res)[:3]:
        show(k, res[k])
    print('rank correlation between the two sets of scores:', round(res['rank_correlation_of_scores'], 3))
    json.dump(res, open(os.path.join(OUT, 'fresh_bridge.json'), 'w'), indent=2)


def fresh(art):
    xb = pd.read_csv(os.path.join(FRESH, 'xbrl_firm_years.csv'))
    s = score(xb, art, PRIMARY + LATE)
    lab = fresh_labels()
    tier = np.array([lab.get((int(c), int(f)), '') for c, f in zip(s['gvkey'], s['fyear'])])
    y = (tier != '').astype(int)
    found = {k for k in lab if k[1] in PRIMARY + LATE}
    seen = set(zip(s['gvkey'][y == 1].astype(int), s['fyear'][y == 1].astype(int)))
    print(f'labelled firm-years 2015-2020: {len(found)}, present in the universe: {len(seen)}')
    print('labelled but absent:', sorted(found - seen))
    s.assign(misstate=y, tier=tier).to_csv(os.path.join(OUT, 'fresh_predictions.csv'), index=False)
    res = {'labelled_absent': sorted([list(k) for k in found - seen])}
    res['primary'] = summarise(s, y, PRIMARY)
    show('PRIMARY 2015-2018', res['primary'])
    res['late_labels_incomplete'] = summarise(s, y, LATE)
    show('2019-2020 (labels incomplete)', res['late_labels_incomplete'])
    keepA = tier != 'B'
    res['tier_A_only'] = summarise(s[keepA].reset_index(drop=True), y[keepA], PRIMARY)
    show('1. tier A only', res['tier_A_only'])
    priced = (s['prcc_f'].notna() & s['csho'].notna()).values
    res['with_price'] = summarise(s[priced].reset_index(drop=True), y[priced], PRIMARY)
    show('2. filer-years with a price', res['with_price'])
    big = (s['at'] >= 10).values
    res['assets_10m'] = summarise(s[big].reset_index(drop=True), y[big], PRIMARY)
    show('3. total assets >= $10m', res['assets_10m'])
    for m in art['ensemble_members'] + ['TT']:
        res[f'member_{m}'] = summarise(s, y, PRIMARY, col=m)
        show(f'4. {m} alone', res[f'member_{m}'])
    q = {(c, f) for c, fs in QUARTER_ONLY.values() for f in fs}
    yq = np.maximum(y, np.array([(int(c), int(f)) in q for c, f in zip(s['gvkey'], s['fyear'])]).astype(int))
    res['with_quarter_only'] = summarise(s, yq, PRIMARY)
    show('5. quarter-only cases added', res['with_quarter_only'])
    auc = res['primary'].get('mean_yearly', {}).get('AUC')
    lo = res['primary'].get('mean_yearly_AUC_CI95', [None])[0]
    res['verdict'] = ('no year with 15 frauds' if auc is None else 'Holds' if auc >= 0.725 else
                      'Weaker but real' if lo > 0.5 else 'Fails')
    print('VERDICT (PROTOCOL_FRESH.md):', res['verdict'])
    json.dump(res, open(os.path.join(OUT, 'fresh_test.json'), 'w'), indent=2)


if __name__ == '__main__':
    art = joblib.load(os.path.join(ROOT, 'results', 'final_models.joblib'))
    {'bridge': bridge, 'fresh': fresh}[sys.argv[1]](art)

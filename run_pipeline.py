"""End-to-end pipeline: improved accounting-fraud detection (extends Bao et al., JAR 2020).

    python run_pipeline.py                  # full run: test years 2003-2014   (~30-45 min, CPU only)
    python run_pipeline.py --fast           # quick run: test years 2003-2008  (~10-15 min)
    python run_pipeline.py --trim_frac 0.3  # optional: keep only 30% of NON-fraud firms for training

Protocol = the authors' own (see run_RUSBoost.m), so numbers are directly comparable:
    * for each test year T in 2003..2014: train on fiscal years 1991..T-2 (2-year gap), test on T
    * serial-fraud rule: training rows from an AAER case that is also in the test year -> label 0
    * metrics are computed per test year and averaged (AUC, NDCG@k, sensitivity@k)
    * hyper-parameters are tuned ONLY on pre-2003 data with rolling-origin CV (no test peeking)
    * test rows are never trimmed or subsampled
    * the Winsorizer is refit on the training rows of every CV fold / test year / final window
    * years with < --min_frauds frauds are listed but left out of period means and bootstraps
    * the deployable model is trained on 1991..--final_train_end (recent AAERs are not revealed yet)
"""
import argparse
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from evaluate import (BOOT_METRICS, bootstrap_period, decision_metrics, period_mean,  # noqa: E402
                      per_year_metrics)
from explain import explain_all  # noqa: E402
from features import Winsorizer, engineer  # noqa: E402
from models import (BalancedBagger, LogReg, RUSBoost, cv_predict, ensemble_scores,  # noqa: E402
                    lgbm_factory, make_folds, mask_serial, random_search, sample_lgbm,
                    sample_xgb, xgb_factory)

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='data/data_FraudDetection_JAR2020.csv')
ap.add_argument('--out', default='results')
ap.add_argument('--fast', action='store_true')
ap.add_argument('--trim_frac', type=float, default=1.0,
                help='fraction of NON-fraud firms kept in training/CV (test is never trimmed)')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--n_jobs', type=int, default=-1)
ap.add_argument('--reuse', default='', help='dir with tuning_*.csv from an earlier run: skip steps 2-4')
ap.add_argument('--skip_final', action='store_true',
                help='do not save the deployable model (it is still fitted for SHAP)')
ap.add_argument('--final_train_end', type=int, default=2010,
                help='last fiscal year of the deployable model (label delay: later AAERs not yet revealed)')
ap.add_argument('--min_frauds', type=int, default=15,
                help='test years with fewer frauds are excluded from period means / bootstraps')
ap.add_argument('--n_boot', type=int, default=1000)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
T0 = time.time()
LOG = open(os.path.join(args.out, 'log.txt'), 'w')


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + '\n')
    LOG.flush()


N_TRIALS = 8 if args.fast else 20
N_BAGS_TUNE = 3
N_BAGS_EVAL = 5 if args.fast else 10
TEST_YEARS = list(range(2003, 2009 if args.fast else 2015))
BASE = 'Baseline: RUSBoost (paper, raw 28)'
PROP = 'PROPOSED: ensemble'
E = args.final_train_end

# ----------------------------------------------------------------------------- data
log('[1/7] loading + feature engineering')
raw = pd.read_csv(args.data)
df, groups = engineer(raw)
df = df[df['fyear'] >= 1991].reset_index(drop=True)
years, y, PAAER = df['fyear'].values, df['misstate'].values, df['p_aaer'].values
GVKEY = df['gvkey'].values
POS_KEY = np.where(np.isnan(PAAER), -GVKEY, PAAER)            # bootstrap cluster: AAER case, else firm
pre_test = np.where((years >= 1991) & (years <= 2001))[0]     # data used for tuning
log(f'   rows={len(df):,}  frauds={y.sum()}  pre-test rows={len(pre_test):,} '
    f'(frauds {y[pre_test].sum()})  test years={TEST_YEARS[0]}-{TEST_YEARS[-1]}')

keep = np.ones(len(df), dtype=bool)                            # optional trimming of non-fraud firms
if args.trim_frac < 1.0:
    rng = np.random.RandomState(args.seed)
    fraud_firms = set(df.loc[df.misstate == 1, 'gvkey'])
    other = np.array(sorted(set(df['gvkey']) - fraud_firms))
    kept = set(rng.choice(other, int(args.trim_frac * len(other)), replace=False)) | fraud_firms
    keep = df['gvkey'].isin(kept).values
    log(f'   TRIM: keeping {args.trim_frac:.0%} of non-fraud firms for training/CV')

CANDIDATES = ['enhanced', 'rank_only', 'enhanced_rank']       # scale-free sets for the proposed model
WCOLS = groups['raw28'] + groups['enhanced_rank']
DW = df[WCOLS].astype(np.float32)


def winsorize(fit_rows):
    """Winsorizer fit on one training window only, applied to all rows."""
    wz = Winsorizer().fit(DW.iloc[fit_rows])
    return wz, wz.transform(DW)


folds = [(tr[keep[tr]], va[keep[va]]) for tr, va in make_folds(years)]
FOLD_D = [winsorize(tr)[1] for tr, _ in folds]


def fmat(cols):
    """Per-fold feature matrices, each winsorised on its own fold's training rows."""
    return [d[cols].values for d in FOLD_D]


def pick_fs(fs_df):
    c = fs_df[fs_df.feature_set.isin(CANDIDATES)]
    assert len(c), f'no scale-free candidate {CANDIDATES} in feature_set_selection.csv'
    return c.sort_values('cv_auc', ascending=False, kind='stable').iloc[0]['feature_set']


def pick_combo(combo_auc):
    """The baseline model (RUSraw) must not be a member of the proposed ensemble; among candidates
    within 0.002 CV-AUC of the best, prefer the simplest (fewest members), then the higher AUC."""
    cand = {k: v for k, v in combo_auc.items() if '+' in k and 'RUSraw' not in k}
    top = max(cand.values())
    return sorted([k for k, v in cand.items() if v >= top - 0.002], key=lambda k: (len(k.split('+')), -cand[k]))[0]


DEFAULT_L = dict(n_estimators=250, learning_rate=0.03, num_leaves=8, min_child_samples=30,
                 colsample_bytree=0.7, subsample=0.8, reg_lambda=3.0, ratio=10)


def build_lgbm(p, n_bags):
    return BalancedBagger(lgbm_factory(p, args.n_jobs), n_bags=n_bags, ratio=p['ratio'], seed=args.seed)


def build_xgb(p, n_bags):   # seed + 1: XGB bags draw different non-fraud samples than LGBM bags
    return BalancedBagger(xgb_factory(p, args.n_jobs), n_bags=n_bags, ratio=p['ratio'], seed=args.seed + 1)


REUSE = bool(args.reuse)
if REUSE:
    def _best(fn):
        h = pd.read_csv(os.path.join(args.reuse, fn)).sort_values('cv_auc', ascending=False, kind='stable')
        h = h.iloc[0].to_dict()
        h.pop('cv_auc')
        return {k: (int(v) if k in ('n_estimators', 'num_leaves', 'min_child_samples', 'max_depth', 'ratio') else float(v))
                for k, v in h.items()}
    best_fs = pick_fs(pd.read_csv(os.path.join(args.reuse, 'feature_set_selection.csv')))
    FEATS = groups[best_fs]
    bestL, bestX = _best('tuning_lgbm.csv'), _best('tuning_xgb.csv')
    es = pd.read_csv(os.path.join(args.reuse, 'ensemble_selection.csv'))
    combo_auc = dict(zip(es.combo, es.cv_auc.astype(float)))
    best_combo = pick_combo(combo_auc)
    ENS_MEMBERS = best_combo.split('+')
    log(f'[reuse] feature set={best_fs}, ensemble={best_combo}, lgbm={bestL}, xgb={bestX}')
else:
    # ----------------------------------------------------------------------------- feature set
    log('[2/7] choosing feature set by validation AUC (default LightGBM, rolling-origin CV)')
    fs_rows = []
    for name in ['raw28'] + CANDIDATES:                         # raw28 = reference only, never chosen
        _, auc = cv_predict(lambda a, b, c: build_lgbm(DEFAULT_L, N_BAGS_TUNE).fit(a, b).predict_proba(c),
                            fmat(groups[name]), y, folds, PAAER)
        fs_rows.append({'feature_set': name, 'n_features': len(groups[name]), 'cv_auc': auc})
        log(f'   {name:<13} n={len(groups[name]):>3}  cv AUC={auc:.4f}')
    fs_df = pd.DataFrame(fs_rows)
    fs_df.to_csv(os.path.join(args.out, 'feature_set_selection.csv'), index=False)
    best_fs = pick_fs(fs_df)
    FEATS = groups[best_fs]
    X_cv, Xraw_cv = fmat(FEATS), fmat(groups['raw28'])
    log(f'   -> using feature set: {best_fs} ({len(FEATS)} features)')

    # ----------------------------------------------------------------------------- tuning
    log('[3/7] tuning LightGBM (rolling-origin CV, serial-fraud masking)')
    bestL, aucL, histL = random_search(sample_lgbm, build_lgbm, X_cv, y, folds, N_TRIALS,
                                       seed=args.seed, n_bags=N_BAGS_TUNE, paaer=PAAER, log=log)
    histL.to_csv(os.path.join(args.out, 'tuning_lgbm.csv'), index=False)
    log('[3/7] tuning XGBoost')
    bestX, aucX, histX = random_search(sample_xgb, build_xgb, X_cv, y, folds, N_TRIALS,
                                       seed=args.seed + 1, n_bags=N_BAGS_TUNE, paaer=PAAER, log=log)
    histX.to_csv(os.path.join(args.out, 'tuning_xgb.csv'), index=False)

    # ----------------------------------------------------------------------------- ensemble choice
    log('[4/7] choosing ensemble recipe on validation folds')
    cv = {
        'LGBM': cv_predict(lambda a, b, c: build_lgbm(bestL, N_BAGS_TUNE).fit(a, b).predict_proba(c),
                           X_cv, y, folds, PAAER),
        'XGB': cv_predict(lambda a, b, c: build_xgb(bestX, N_BAGS_TUNE).fit(a, b).predict_proba(c),
                          X_cv, y, folds, PAAER),
        'LR': cv_predict(lambda a, b, c: LogReg().fit(a, b).predict_proba(c), X_cv, y, folds, PAAER),
        'RUSenh': cv_predict(lambda a, b, c: RUSBoost(seed=args.seed).fit(a, b).predict_proba(c),
                             X_cv, y, folds, PAAER),
        'RUSraw': cv_predict(lambda a, b, c: RUSBoost(seed=args.seed).fit(a, b).predict_proba(c),
                             Xraw_cv, y, folds, PAAER),
    }
    combo_auc = {k: v[1] for k, v in cv.items()}
    for k, a in combo_auc.items():
        log(f'   single model {k:<7} cv AUC={a:.4f}')
    combos = ['LGBM+XGB', 'LGBM+XGB+LR', 'LGBM+XGB+RUSenh', 'LGBM+XGB+RUSraw',
              'LGBM+XGB+LR+RUSenh', 'LGBM+XGB+LR+RUSenh+RUSraw']
    for c in combos:
        mem = c.split('+')
        aucs = [roc_auc_score(y[cv['LGBM'][0][k][0]],
                              ensemble_scores([cv[m][0][k][1] for m in mem])) for k in range(len(folds))]
        combo_auc[c] = float(np.mean(aucs))
        log(f'   ensemble {c:<28} cv AUC={combo_auc[c]:.4f}')
    best_combo = pick_combo(combo_auc)
    log(f'   -> ensemble = {best_combo}')
    pd.DataFrame([{'combo': k, 'cv_auc': v} for k, v in combo_auc.items()]).to_csv(
        os.path.join(args.out, 'ensemble_selection.csv'), index=False)
    ENS_MEMBERS = best_combo.split('+')

# ----------------------------------------------------------------------------- expanding-window test
log(f'[5/7] expanding-window evaluation, test years {TEST_YEARS[0]}-{TEST_YEARS[-1]}')
MODEL_NAMES = [BASE, 'LightGBM bagged (raw 28)', 'RUSBoost (enhanced)', 'LightGBM bagged (enhanced)',
               'XGBoost bagged (enhanced)', 'Logistic regression (enhanced)', PROP]
store = {m: [] for m in MODEL_NAMES}
te_rows = []
for yt in TEST_YEARS:
    t0 = time.time()
    tr = np.where((years >= 1991) & (years <= yt - 2) & keep)[0]
    te = np.where(years == yt)[0]
    ytr = mask_serial(y, PAAER, tr, te)
    _, D = winsorize(tr)
    X_all, X_raw = D[FEATS].values, D[groups['raw28']].values
    p = {}
    p[BASE] = RUSBoost(seed=args.seed).fit(X_raw[tr], ytr).predict_proba(X_raw[te])
    p['LightGBM bagged (raw 28)'] = build_lgbm(bestL, N_BAGS_EVAL).fit(X_raw[tr], ytr).predict_proba(X_raw[te])
    p['RUSBoost (enhanced)'] = RUSBoost(seed=args.seed).fit(X_all[tr], ytr).predict_proba(X_all[te])
    p['LightGBM bagged (enhanced)'] = build_lgbm(bestL, N_BAGS_EVAL).fit(X_all[tr], ytr).predict_proba(X_all[te])
    p['XGBoost bagged (enhanced)'] = build_xgb(bestX, N_BAGS_EVAL).fit(X_all[tr], ytr).predict_proba(X_all[te])
    p['Logistic regression (enhanced)'] = LogReg().fit(X_all[tr], ytr).predict_proba(X_all[te])
    key = {'LGBM': 'LightGBM bagged (enhanced)', 'XGB': 'XGBoost bagged (enhanced)',
           'LR': 'Logistic regression (enhanced)', 'RUSenh': 'RUSBoost (enhanced)', 'RUSraw': BASE}
    p[PROP] = ensemble_scores([p[key[m]] for m in ENS_MEMBERS])
    for m in MODEL_NAMES:
        store[m].append(p[m])
    te_rows.append(te)
    a_b, a_p = roc_auc_score(y[te], p[BASE]), roc_auc_score(y[te], p[PROP])
    log(f'   test {yt}: train rows={len(tr):,} (frauds {ytr.sum()})  test frauds={y[te].sum():>3}  '
        f'AUC baseline={a_b:.3f}  proposed={a_p:.3f}  [{time.time() - t0:.0f}s]')

te_all = np.concatenate(te_rows)
yt_all, yrs_all = y[te_all], years[te_all]
PRED = {m: np.concatenate(store[m]) for m in MODEL_NAMES}
out = df.loc[te_all, ['gvkey', 'fyear', 'misstate']].copy()
for m in MODEL_NAMES:
    out[m] = PRED[m]
out.to_csv(os.path.join(args.out, 'test_predictions.csv'), index=False)

# ----------------------------------------------------------------------------- results
log('[6/7] results')
periods = [(2003, 2008)] + ([] if args.fast else [(2003, 2014)])
yearly = {m: per_year_metrics(yt_all, PRED[m], yrs_all) for m in MODEL_NAMES}
pd.concat({m: d for m, d in yearly.items()}, names=['model']).reset_index(level=0).to_csv(
    os.path.join(args.out, 'yearly_metrics.csv'), index=False)
dm = pd.concat({m: decision_metrics(yt_all, PRED[m], yrs_all) for m in MODEL_NAMES},
               names=['model']).reset_index(level=0)
dm.to_csv(os.path.join(args.out, 'decision_metrics.csv'), index=False)
dm.loc[dm.k == '1%', ['model', 'year', 'n', 'frauds', 'investigated', 'TP', 'FP', 'FN', 'TN']].to_csv(
    os.path.join(args.out, 'confusion_top1pct.csv'), index=False)

n_fr = pd.Series(yt_all).groupby(yrs_all).sum()
excluded = [int(t) for t in TEST_YEARS if n_fr.get(t, 0) < max(1, args.min_frauds)]
log(f'   years with < {args.min_frauds} frauds, listed in yearly_metrics.csv but excluded from period '
    f'means and bootstraps: {excluded or "none"}')

summary = {'feature_set': best_fs, 'ensemble': best_combo, 'best_lgbm': bestL, 'best_xgb': bestX,
           'cv_auc': {k: float(v) for k, v in combo_auc.items()}, 'final_train_end': E,
           'min_frauds': args.min_frauds, 'excluded_years': excluded, 'n_boot': args.n_boot, 'periods': {}}
for lo, hi in periods:
    use = [t for t in range(lo, hi + 1) if t in TEST_YEARS and t not in excluded]
    boot = bootstrap_period(yt_all, PRED, yrs_all, POS_KEY[te_all], GVKEY[te_all], use,
                            n_boot=args.n_boot, seed=args.seed)
    means = {m: period_mean(yearly[m], lo, hi, args.min_frauds) for m in MODEL_NAMES}
    rows = []
    for m in MODEL_NAMES:
        r = {'model': m, **means[m].to_dict(),
             'AUC_CI95': f"[{np.percentile(boot[m][:, 0], 2.5):.3f}, {np.percentile(boot[m][:, 0], 97.5):.3f}]"}
        for j, met in enumerate(BOOT_METRICS):
            d = boot[m][:, j] - boot[BASE][:, j]
            r[f'{met}_gain'] = means[m][met] - means[BASE][met]          # point estimate
            r[f'{met}_gain_CI95'] = f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"
            r[f'P({met}_gain<=0)'] = float((d <= 0).mean())
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.out, f'test_metrics_{lo}_{hi}.csv'), index=False)
    log(f'\n=== test years {lo}-{hi} (mean of yearly metrics over {use}) ===')
    log(tab.round(4).to_string(index=False))
    base_rate = 100 * np.mean([yt_all[yrs_all == t].mean() for t in use])
    log(f'   at a 1% investigation budget: proposed finds {100 * means[PROP]["Prec@1%"]:.1f} fraud cases per '
        f'100 investigations, baseline {100 * means[BASE]["Prec@1%"]:.1f}, random picking {base_rate:.1f}')
    summary['periods'][f'{lo}-{hi}'] = {'years_used': use, **tab.set_index('model').to_dict('index')}

# calibration: proposed out-of-sample scores (years <= final_train_end) -> fraud probability
cal = yrs_all <= E
assert cal.any(), f'no test years <= --final_train_end {E} to calibrate on'
iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(PRED[PROP][cal], yt_all[cal])
cdf = pd.DataFrame({'score': PRED[PROP][cal], 'fraud_probability': iso.predict(PRED[PROP][cal]),
                    'fraud': yt_all[cal]})
cdf['bin'] = pd.qcut(cdf['score'], 10, labels=False, duplicates='drop')
calib = cdf.groupby('bin').agg(mean_score=('score', 'mean'), mean_fraud_probability=('fraud_probability', 'mean'),
                               observed_fraud_rate=('fraud', 'mean'), n=('fraud', 'size')).reset_index()
calib.to_csv(os.path.join(args.out, 'calibration.csv'), index=False)
cal_years = sorted({int(t) for t in yrs_all[cal]})
log(f'\n   isotonic calibration fit on proposed test-year scores {cal_years[0]}-{cal_years[-1]}:')
log(calib.round(4).to_string(index=False))
summary['calibration_years'] = cal_years
json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=2, default=float)

# figures: yearly AUC lines + pooled ROC/PR (2003-2008)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for m in MODEL_NAMES:
    ax[0].plot(yearly[m]['year'], yearly[m]['AUC'], marker='o', ms=3,
               lw=2.6 if m == PROP else (2.0 if m == BASE else 0.9), ls='--' if m == BASE else '-', label=m)
ax[0].set(xlabel='Test year', ylabel='AUC', title='AUC by test year')
ax[0].legend(fontsize=6.5)
msk = yrs_all <= 2008
for m in MODEL_NAMES:
    fpr, tpr, _ = roc_curve(yt_all[msk], PRED[m][msk])
    ax[1].plot(fpr, tpr, lw=2.4 if m == PROP else 1.0)
    pr, rc, _ = precision_recall_curve(yt_all[msk], PRED[m][msk])
    ax[2].plot(rc, pr, lw=2.4 if m == PROP else 1.0)
ax[1].plot([0, 1], [0, 1], 'k:', lw=0.8)
ax[1].set(xlabel='FPR', ylabel='TPR', title='ROC, pooled test 2003-2008')
ax[2].set(xlabel='Recall', ylabel='Precision', title='Precision-recall, pooled test 2003-2008')
plt.tight_layout()
plt.savefig(os.path.join(args.out, 'performance_curves.png'), dpi=150)
plt.close()

# ----------------------------------------------------------------------------- deployable model + SHAP
log(f'\n[7/7] deployable model: fiscal years 1991-{E} (label delay), ensemble members {ENS_MEMBERS}')
tr = np.where((years >= 1991) & (years <= E) & keep)[0]
wz, D = winsorize(tr)
X_all = D[FEATS].values
makers = {'LGBM': lambda: build_lgbm(bestL, 10).fit(X_all[tr], y[tr]),
          'XGB': lambda: build_xgb(bestX, 10).fit(X_all[tr], y[tr]),
          'LR': lambda: LogReg().fit(X_all[tr], y[tr]),
          'RUSenh': lambda: RUSBoost(seed=args.seed).fit(X_all[tr], y[tr])}
mem = {m: makers[m]() for m in ENS_MEMBERS}
if not args.skip_final:
    joblib.dump({'features': FEATS, 'winsorizer': wz, 'winsor_cols': WCOLS, 'ensemble_members': ENS_MEMBERS,
                 'models': mem, 'calibrator': iso, 'train_years': [1991, E]},
                os.path.join(args.out, 'final_models.joblib'), compress=3)

ex_years = list(range(E + 2, TEST_YEARS[-1] + 1))
note = ''
if not ex_years:
    ex_years, note = list(range(2003, 2009)), ' (no test year after the training window: in-sample for this model)'
log(f'   SHAP on the deployed members, explained on fiscal years {ex_years[0]}-{ex_years[-1]}{note}')
ex = np.where(np.isin(years, ex_years))[0]
X_ex = X_all[ex]
raw_ex = {m: mem[m].predict_proba(X_ex) for m in ENS_MEMBERS}
pct = np.zeros(len(ex))
for t in ex_years:                     # same recipe as scoring: rank each member within the fiscal year
    k = years[ex] == t
    pct[k] = ensemble_scores([raw_ex[m][k] for m in ENS_MEMBERS])
scores = pd.DataFrame({'risk_percentile': pct, 'risk_score': iso.predict(pct)})
imp = explain_all(mem, pd.DataFrame(X_ex, columns=FEATS), y[ex],
                  df.loc[ex, ['gvkey', 'fyear']].reset_index(drop=True), scores,
                  os.path.join(args.out, 'explain'), seed=args.seed, log=log)
log('\n   Top-10 features by mean |SHAP|:')
log(imp.head(10)[['feature', 'mean_abs_shap', 'meaning']].to_string(index=False))
log(f'\nDone in {(time.time() - T0) / 60:.1f} min. Outputs in {args.out}/')

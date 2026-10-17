# Explainable accounting-fraud detection (extension of Bao et al., JAR 2020)

Minor project: extends the RUSBoost baseline of *Detecting Accounting Fraud in Publicly Traded
U.S. Firms Using a Machine Learning Approach* (Bao, Ke, Li, Yu, Zhang 2020) with scale-free
accounting features, tuned gradient-boosting ensembles, decision metrics at an investigation
budget, calibrated scores, and SHAP explanations of the deployed model.

## Setup
```
pip install -r requirements.txt          # exact pins: the saved model only loads with these versions
# put data_FraudDetection_JAR2020.csv (from github.com/JarFraud/FraudDetection) in data/
python run_pipeline.py --fast            # test years 2003-2008, ~12 min on a CPU
python run_pipeline.py                   # test years 2003-2014, ~20 min
python run_pipeline.py --reuse results --out results2   # reuse saved tuning, skip steps 2-4
python score_new_data.py --input firms.csv --year 2012 --out scored.csv
python scripts/smoke_score.py            # 20-firm end-to-end check of the scorer
```
Flags: `--final_train_end 2010` (last fiscal year the deployable model trains on), `--min_frauds 15`
(test years with fewer labelled frauds are listed but excluded from averages and bootstraps),
`--n_boot 1000`, `--trim_frac`, `--seed`.

## What is different from the paper
| Area | Paper | This project |
|---|---|---|
| Features | 28 raw Compustat items | **Scale-free only**: 14 paper ratios + Beneish/Altman parts, common-size ratios, growth gaps, and their within-year percentile ranks (`enhanced_rank`, 133 features, chosen on validation years). The raw dollar items are used by the baseline only. |
| Model | RUSBoost | Balanced-bagged LightGBM + XGBoost + logistic regression, rank-averaged per year. RUSBoost re-implemented from the authors' MATLAB settings (300 trees, min leaf 5, learning rate 0.1, 1:1 under-sampling, MATLAB's boosting default of 10 splits per tree). |
| Winsorising | – | Refit on each expanding training window (no fixed 1991-2001 caps leaking into later years) |
| Tuning | tree count only | random search on rolling-origin CV using pre-2003 data only |
| Decision step | ranking only | precision / recall / lift at a 1%, 2%, 5% investigation budget + confusion matrix at 1%; isotonic-calibrated `fraud_probability` |
| Significance | – | paired bootstrap (1000 resamples) clustered by AAER case and by firm, for AUC, NDCG@5% and Sens@5% |
| Deployable model | – | trained on 1991-2010 only, because AAER labels for later years are still incomplete (test-year frauds fall from 69 in 2003 to 4 in 2014) |
| Explainability | none | SHAP on the **deployed** ensemble members (all bags), on a real-base-rate sample of 2012-2014 |
| Protocol | expanding window, 2-yr gap, serial-fraud masking | identical |

## Results (full run, `results/`; mean of yearly metrics; 2013-2014 excluded, they have 11 and 4 frauds)
Test years 2003-2012:

| Model | AUC | NDCG@5% | Sens@5% | Prec@1% | Lift@1% |
|---|---|---|---|---|---|
| Baseline RUSBoost (raw 28) | 0.705 | 0.066 | 0.098 | 1.8% | 3.1 |
| RUSBoost (enhanced features) | 0.732 | 0.071 | 0.127 | 1.6% | 2.7 |
| LightGBM bagged (enhanced) | 0.730 | 0.074 | 0.142 | 1.9% | 3.5 |
| **Proposed ensemble (LGBM+XGB+LR, enhanced)** | **0.732** | **0.081** | **0.155** | **1.9%** | **3.5** |

Gain of the proposed ensemble over the baseline (paired clustered bootstrap, 95% CI):

| Metric | Gain | 95% CI | P(gain ≤ 0) |
|---|---|---|---|
| AUC | +0.027 | [+0.007, +0.045] | 0.002 |
| NDCG@5% | +0.015 | [−0.012, +0.042] | 0.135 |
| Sens@5% | +0.057 | [+0.013, +0.095] | 0.004 |

Test years 2003-2008 only: baseline AUC 0.705 / Sens@5% 0.090, proposed 0.727 / 0.164
(AUC gain +0.021 [+0.000, +0.044]; Sens@5% gain +0.074 [+0.024, +0.121]).

**Read these honestly:**
* At a 1% investigation budget the proposed model finds about **1.9 fraud cases per 100
  investigations**, versus 1.8 for the baseline and 0.6 for random picking. The base rate is
  about 0.6% of firm-years. So roughly 98 of every 100 flagged firms are NOT (labelled) fraud:
  this is a tool for prioritising investigations, not a fraud classifier. Per-year confusion
  matrices at the 1% budget are in `results/confusion_top1pct.csv`; several years have 0 hits.
* The AUC and Sens@5% gains over the paper's RUSBoost are statistically significant; the NDCG@5%
  gain is not. RUSBoost on the enhanced features alone reaches the same AUC (0.732), so most of
  the gain comes from the features, not the ensemble.
* The baseline here uses MATLAB's default of 10 splits per boosting tree. An earlier version of
  this repo grew unlimited-depth trees and scored higher (AUC 0.724 on 2003-2014); against that
  variant the ensemble's AUC gain was not significant. The paper's own reported baseline is
  slightly stronger than either at the very top of the ranking (NDCG@1%).
* Validation CV AUC (pre-2003): raw 28 → 0.690, enhanced_rank ensemble → 0.760. Selection is
  made on only 3 validation years with ~300 frauds, so differences of ~0.005 between candidate
  feature sets / ensembles are within noise.
* `fraud_probability` is an isotonic calibration of the rank score, fit on out-of-sample
  2003-2010 scores (`results/calibration.csv`; top decile observed rate 2.0%, predicted 2.1%).
  It is a step function and saturates near 2-3%.

## Explainability findings (SHAP on the deployed LGBM+XGB+LR members, 2012-2014, real base rate)
Top drivers: firm size (`log_at`, `log_sale` and their ranks), gross PPE / assets, depreciation /
assets, external financing need, investments / assets, current ratio, cash / assets.
**Firm size still dominates** even without raw dollar items. Dropping `log_at`/`log_sale` from the
feature set would make the model size-neutral; it was kept because the paper's own features
carry size. See `results/explain/` (bar, beeswarm, dependence plots, waterfalls,
`top_flagged_explained.csv` with `risk_percentile` + calibrated `risk_score`).
Caveat: SHAP explains what the model uses, not why a firm committed fraud; labels are SEC-enforced
cases only (undetected fraud is labelled 0, and 2011+ labels are still being discovered).

## Scoring new firms
`score_new_data.py` ranks the input firms against the **full market cross-section of the same
fiscal year** taken from `--reference` (the training CSV by default), because 66 of the features
and the final score are within-year ranks: scoring 10 firms on their own would rank them only
against each other. Include each firm's prior-year row so lag features exist; only the `--year`
rows of the input firms are written out, with `raw_score`, `risk_percentile`, `fraud_probability`
and `market_size`.

## Known limitations
* Beneish components are approximations (SGAI fixed at 1, gross PPE instead of net, accruals from
  working capital). Altman Z uses the paper's EBIT ratio.
* 3 validation years for all model selection; 2013-2014 too sparse to evaluate.
* Labels are enforcement actions, so "clean" firms include undetected fraud.

## Files
`run_pipeline.py` main script | `src/` features, models, evaluation, SHAP | `score_new_data.py` |
`scripts/smoke_score.py` | `results/` metrics, decision metrics, calibration, predictions, plots,
saved model (`final_models.joblib`, git-ignored; regenerate with `run_pipeline.py`)

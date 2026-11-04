# Explainable accounting-fraud detection (extension of Bao et al., JAR 2020)

Minor project: extends the RUSBoost baseline of *Detecting Accounting Fraud in Publicly Traded
U.S. Firms Using a Machine Learning Approach* (Bao, Ke, Li, Yu, Zhang 2020) with scale-free
accounting features, tuned gradient-boosting ensembles, decision metrics at an investigation
budget, calibrated scores, and SHAP explanations of the deployed model.

## Setup
```
pip install -r requirements.txt          # exact pins: the saved model only loads with these versions
# put data_FraudDetection_JAR2020.csv (from github.com/JarFraud/FraudDetection) in data/
python run_pipeline.py --fast            # test years 2003-2008, ~30-45 min on a CPU
python run_pipeline.py                   # test years 2003-2014, ~2 hours
python run_pipeline.py --reuse results --out results2   # reuse saved tuning, skip steps 2-4 (90 min)
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
| Model | RUSBoost | Balanced-bagged LightGBM + XGBoost + logistic regression + a **Temporal Transformer** over each firm's last 5 fiscal years, rank-averaged per year. RUSBoost re-implemented from the authors' MATLAB settings (300 trees, min leaf 5, learning rate 0.1, 1:1 under-sampling, MATLAB's boosting default of 10 splits per tree). |
| Winsorising | – | Refit on each expanding training window (no fixed 1991-2001 caps leaking into later years) |
| Tuning | tree count only | random search on rolling-origin CV using pre-2003 data only |
| Decision step | ranking only | precision / recall / lift at a 1%, 2%, 5% investigation budget + confusion matrix at 1%; isotonic-calibrated `fraud_probability` |
| Significance | – | paired bootstrap (1000 resamples) clustered by AAER case and by firm, for AUC, NDCG@5% and Sens@5% |
| Deployable model | – | trained on 1991-2010 only, because AAER labels for later years are still incomplete (test-year frauds fall from 69 in 2003 to 4 in 2014) |
| Explainability | none | SHAP on the **deployed** tree and LR members (all bags), on a real-base-rate sample of 2012-2014. The Temporal Transformer is not explained. |
| Protocol | expanding window, 2-yr gap, serial-fraud masking | identical |

## Results (full run, `results/`; mean of yearly metrics; 2013-2014 excluded, they have 11 and 4 frauds)
Test years 2003-2012:

| Model | AUC | NDCG@5% | Sens@5% | Prec@1% | Lift@1% |
|---|---|---|---|---|---|
| Baseline RUSBoost (raw 28) | 0.705 | 0.066 | 0.098 | 1.8% | 3.1 |
| RUSBoost (enhanced features) | 0.732 | 0.071 | 0.127 | 1.6% | 2.7 |
| LightGBM bagged (enhanced) | 0.730 | 0.074 | 0.142 | 1.9% | 3.5 |
| Temporal Transformer alone | 0.721 | 0.056 | 0.103 | 1.0% | 1.4 |
| Ensemble without firm history (LGBM+XGB+LR) | 0.735 | 0.082 | 0.156 | 1.8% | 3.3 |
| **Proposed ensemble (LGBM+XGB+LR+Temporal Transformer)** | **0.743** | **0.084** | **0.162** | **2.2%** | **3.3** |

Gain of the proposed ensemble over the baseline (paired clustered bootstrap, 95% CI):

| Metric | Gain | 95% CI | P(gain ≤ 0) |
|---|---|---|---|
| AUC | +0.039 | [+0.019, +0.057] | 0.000 |
| NDCG@5% | +0.018 | [−0.008, +0.045] | 0.080 |
| Sens@5% | +0.064 | [+0.023, +0.106] | 0.000 |

Test years 2003-2008 only: baseline AUC 0.705 / Sens@5% 0.090, proposed 0.737 / 0.174
(AUC gain +0.032 [+0.011, +0.054]; Sens@5% gain +0.084 [+0.033, +0.127]).

**Read these honestly:**
* At a 1% investigation budget the proposed model finds about **2.2 fraud cases per 100
  investigations**, versus 1.8 for the baseline and 0.6 for random picking. These are seed 42
  figures, and the top of the ranking is noisy: over 3 seeds, adding the Temporal Transformer moves
  Prec@1% from 1.9% to 2.1% but lowers Sens@1% from 3.5% to 2.9%. The base rate is
  about 0.6% of firm-years. So roughly 98 of every 100 flagged firms are NOT (labelled) fraud:
  this is a tool for prioritising investigations, not a fraud classifier. Per-year confusion
  matrices at the 1% budget are in `results/confusion_top1pct.csv`; several years have 0 hits.
* The AUC and Sens@5% gains over the paper's RUSBoost are statistically significant; the NDCG@5%
  gain is not. RUSBoost on the enhanced features alone reaches AUC 0.732, so most of the gain
  comes from the features, not the ensemble.
* The baseline here uses MATLAB's default of 10 splits per boosting tree. An earlier version of
  this repo grew unlimited-depth trees and scored higher (AUC 0.724 on 2003-2014); against that
  variant the ensemble's AUC gain was not significant. The paper's own reported baseline is
  slightly stronger than either at the very top of the ranking (NDCG@1%).
* Validation CV AUC (pre-2003): raw 28 → 0.690, enhanced_rank ensemble → 0.760, with the Temporal
  Transformer → 0.772. Selection is
  made on only 3 validation years with ~300 frauds, so differences of ~0.005 between candidate
  feature sets / ensembles are within noise.
* `fraud_probability` is an isotonic calibration of the rank score, fit on out-of-sample
  2003-2010 scores (`results/calibration.csv`; top decile observed rate 2.1%, predicted 2.2%).
  It is a step function and saturates near 2-3%.

## Model experiments (`experiments/`, rules fixed in advance in `experiments/PROTOCOL.md`)
Each candidate was frozen on pre-2003 folds, then compared with the ensemble without firm history
on test years 2003-2012: 3 seeds, paired clustered bootstrap. To be adopted, a candidate had to gain
at least 0.002 CV AUC, then show a significant test gain that holds on all seeds and in both
2003-2008 and 2009-2012, with no metric harmed.

| Candidate | CV AUC gain | Test AUC | NDCG@5% | Sens@5% | Verdict |
|---|---|---|---|---|---|
| Reference (LGBM+XGB+LR) | – | 0.735 | 0.079 | 0.152 | – |
| **+ Temporal Transformer** | **+0.0126** | **0.743** | **0.082** | **0.159** | **Adopted, "promising, not confirmed"** |
| + Temporal Mamba | +0.0041 | 0.738 | 0.080 | 0.150 | Passed, weaker; not added to the ensemble |
| + FT-Transformer | −0.0036 | 0.728 | 0.073 | 0.136 | Rejected |
| + TabM | −0.0026 | 0.726 | 0.076 | 0.139 | Rejected |
| + Mambular | −0.0011 | 0.729 | 0.069 | 0.130 | Rejected |
| Dynamic RFE (94 features in CV, re-selected per window) | +0.0003 | 0.733 | 0.074 | 0.139 | Rejected |
| Focal PU XGBoost replacing XGBoost | +0.0005 | 0.731 | 0.080 | 0.146 | Rejected |

* The adopted gain is AUC **+0.0084** (95% CI [+0.0036, +0.0135], p = 0.003). It is positive on all
  3 seeds and in both sub-periods, but its Holm-adjusted p is 0.11, which is why it is labelled
  "promising, not confirmed". The NDCG@5% (+0.003) and Sens@5% (+0.007) gains are not significant.
* Only the models that read a firm's history help. Feature-token FT-Transformer, Mambular and TabM
  do not beat the trees on single-year features, in line with recent tabular benchmarks.
* Leakage audit on validation folds (`experiments/audit_temporal.py`): every history entry is the
  same firm in consecutive earlier years; training on labels shuffled within each year gives AUC
  0.53-0.55; the ensemble's gain holds for frauds at firms with no fraud label in training
  (0.757 → 0.769).

## Explainability findings (SHAP on the deployed LGBM+XGB+LR members, 2012-2014, real base rate)
Top drivers: firm size (`log_at`, `log_sale` and their ranks), gross PPE / assets, depreciation /
assets, external financing need, investments / assets, current ratio, cash / assets.
**Firm size still dominates** even without raw dollar items. Dropping `log_at`/`log_sale` from the
feature set would make the model size-neutral; it was kept because the paper's own features
carry size. See `results/explain/` (bar, beeswarm, dependence plots, waterfalls,
`top_flagged_explained.csv` with `risk_percentile` + calibrated `risk_score`).
Caveat: SHAP explains what the model uses, not why a firm committed fraud; labels are SEC-enforced
cases only (undetected fraud is labelled 0, and 2011+ labels are still being discovered).
The Temporal Transformer is one of four ranked members and SHAP does not cover it, so
`top_flagged_explained.csv` pairs the full ensemble's risk score with drivers from the tree members.

## Scoring new firms
`score_new_data.py` ranks the input firms against the **full market cross-section of the same
fiscal year** taken from `--reference` (the training CSV by default), because 66 of the features
and the final score are within-year ranks: scoring 10 firms on their own would rank them only
against each other. Include each firm's prior-year row so lag features exist. The Temporal
Transformer reads the firm's last 5 fiscal years: years missing from the input come from
`--reference`, and a firm in neither file is scored on the shorter history it has. Only the `--year`
rows of the input firms are written out, with `raw_score`, `risk_percentile`, `fraud_probability`
and `market_size`.

## Known limitations
* Beneish components are approximations (SGAI fixed at 1, gross PPE instead of net, accruals from
  working capital). Altman Z uses the paper's EBIT ratio.
* 3 validation years for all model selection; 2013-2014 too sparse to evaluate.
* Labels are enforcement actions, so "clean" firms include undetected fraud.

## Files
`run_pipeline.py` main script | `src/` features, models, deep models, evaluation, SHAP |
`score_new_data.py` | `scripts/smoke_score.py` | `experiments/` candidate tests and protocol, with
outputs in `results/experiments/` | `results/` metrics, decision metrics, calibration, predictions, plots,
saved model (`final_models.joblib`, git-ignored; regenerate with `run_pipeline.py`)

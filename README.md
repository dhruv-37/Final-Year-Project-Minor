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
| Features | 28 raw Compustat items | **Scale-free only**: 14 paper ratios + Beneish/Altman parts, common-size ratios, growth gaps, and their within-year percentile ranks (`enhanced_rank`, 133 features, chosen on validation years), plus 7 within-year ranks of market-value and share-count movement. The raw dollar items are used by the baseline only. |
| Model | RUSBoost | Balanced-bagged LightGBM + XGBoost, whose training rows lose half their weight every 3 years of age, + logistic regression + a **Temporal Transformer** over each firm's last 5 fiscal years, rank-averaged per year. RUSBoost re-implemented from the authors' MATLAB settings (300 trees, min leaf 5, learning rate 0.1, 1:1 under-sampling, MATLAB's boosting default of 10 splits per tree). |
| Winsorising | – | Refit on each expanding training window (no fixed 1991-2001 caps leaking into later years) |
| Tuning | tree count only | random search on rolling-origin CV using pre-2003 data only |
| Decision step | ranking only | precision / recall / lift at a 1%, 2%, 5% investigation budget + confusion matrix at 1%; isotonic-calibrated `fraud_probability` |
| Significance | – | paired bootstrap (1000 resamples) clustered by AAER case and by firm, for AUC, NDCG@5% and Sens@5% |
| Deployable model | – | trained on 1991-2010 only, because AAER labels for later years are still incomplete (test-year frauds fall from 69 in 2003 to 4 in 2014) |
| Explainability | none | SHAP on all four **deployed** members, on a real-base-rate sample of 2012-2014. The Temporal Transformer's explanations were audited: 7 of 8 checks passed, and its global ranking failed the randomisation check, so it is indicative only. |
| Protocol | expanding window, 2-yr gap, serial-fraud masking | identical |

## Results (full run, `results/`; mean of yearly metrics; 2013-2014 excluded, they have 11 and 4 frauds)
Test years 2003-2012:

| Model | AUC | NDCG@5% | Sens@5% | Prec@1% | Lift@1% |
|---|---|---|---|---|---|
| Baseline RUSBoost (raw 28) | 0.705 | 0.066 | 0.098 | 1.8% | 3.1 |
| RUSBoost (enhanced features) | 0.732 | 0.071 | 0.127 | 1.6% | 2.7 |
| LightGBM bagged (enhanced) | 0.730 | 0.074 | 0.142 | 1.9% | 3.5 |
| Temporal Transformer alone | 0.721 | 0.056 | 0.103 | 1.0% | 1.4 |
| Previous ensemble (133 features, no recency weights, LGBM+XGB+LR+Temporal Transformer) | 0.743 | 0.084 | 0.162 | 2.2% | 3.3 |
| Ensemble without firm history (LGBM+XGB+LR) | 0.740 | 0.085 | 0.157 | 1.7% | 2.7 |
| **Proposed ensemble (LGBM+XGB+LR+Temporal Transformer)** | **0.748** | **0.081** | **0.151** | **1.9%** | **2.7** |

Gain of the proposed ensemble over the baseline (paired clustered bootstrap, 95% CI):

| Metric | Gain | 95% CI | P(gain ≤ 0) |
|---|---|---|---|
| AUC | +0.044 | [+0.025, +0.062] | 0.000 |
| NDCG@5% | +0.015 | [−0.011, +0.042] | 0.118 |
| Sens@5% | +0.053 | [+0.013, +0.093] | 0.003 |

Test years 2003-2008 only: baseline AUC 0.705 / Sens@5% 0.090, proposed 0.745 / 0.163
(AUC gain +0.040 [+0.018, +0.061]; Sens@5% gain +0.073 [+0.023, +0.121]).

**Read these honestly:**
* **The AUC stays below 0.8.** The 0.748 above is seed 42 of the pipeline; over 3 seeds the experiment
  gives 0.748 too. This project's 2-year gap and serial-fraud masking are stricter than the evaluations
  behind many higher figures in the literature.
* The latest gain improves the ranking as a whole, not its top. Over 3 seeds, against the previous
  ensemble, AUC rises from 0.743 to 0.748, while Prec@1% falls from 2.1% to 1.9%, Sens@5% moves from
  0.159 to 0.157 and NDCG@5% from 0.082 to 0.084. The last two moves are within noise. If an
  investigator can only review the top 1%, the previous ensemble is at least as good.
* At a 1% investigation budget the proposed model finds about **1.9 fraud cases per 100
  investigations**, versus 1.8 for the baseline and 0.6 for random picking. The top of the ranking
  is noisy. The base rate is about 0.6% of firm-years. So roughly 98 of every 100 flagged firms are
  NOT (labelled) fraud: this is a tool for prioritising investigations, not a fraud classifier.
  Per-year confusion matrices at the 1% budget are in `results/confusion_top1pct.csv`; several years
  have 0 hits.
* The AUC and Sens@5% gains over the paper's RUSBoost are statistically significant; the NDCG@5%
  gain is not. RUSBoost on the enhanced features alone reaches AUC 0.732, so most of the gain
  comes from the features, not the ensemble.
* The baseline here uses MATLAB's default of 10 splits per boosting tree. An earlier version of
  this repo grew unlimited-depth trees and scored higher (AUC 0.724 on 2003-2014); against that
  variant the ensemble's AUC gain was not significant. The paper's own reported baseline is
  slightly stronger than either at the very top of the ranking (NDCG@1%).
* Validation CV AUC (pre-2003): raw 28 → 0.690, enhanced_rank ensemble → 0.760, with the Temporal
  Transformer → 0.772, with market features, recency weights and retuned trees → 0.787. Selection is
  made on only 3 validation years with ~300 frauds, so differences of ~0.005 between candidate
  feature sets / ensembles are within noise.
* `fraud_probability` is an isotonic calibration of the rank score, fit on out-of-sample
  2003-2010 scores (`results/calibration.csv`; top decile observed rate 2.2%, predicted 2.2%).
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

### Dynamic feature selection: tried, did not raise ROC
The 133 features are a fixed candidate pool. To test whether choosing features dynamically helps, 10
selection families were run inside every training window, under rules fixed in advance
(`experiments/PROTOCOL_FS.md`), against the current ensemble. A family's best setting had to gain at
least +0.002 validation AUC on fresh seeds before it could see the test years. None did, so the test
years were not used and the pipeline keeps a fixed pool.

| Family | Fresh-seed CV AUC gain |
|---|---|
| Pool of 133 + 60 firm-history features, then DRFE | +0.0008 |
| DRFE for logistic regression only | −0.0002 |
| Soft selection: each bag samples features by importance | −0.0018 |
| Per-firm-year gates inside the Temporal Transformer | −0.0025 |
| mRMR | −0.0026 |
| DRFE-subset LightGBM as an extra member | −0.0041 |
| Adversarial drift filter | −0.0047 |
| Out-of-time permutation importance | −0.0067 |
| Stability selection | −0.0207 |
| Shadow-feature test (Boruta) | −0.0311 |

These 10 families are what was tried, not every method that exists. The code for them and for the
other rejected methods was removed after testing; it is in git history at commit `255e803`, and
their scores remain in `results/experiments/`.

**Is it at least no worse?** A second test, [Protocol 6](experiments/PROTOCOL_NI.md), asked a different
question: can selection run inside every window without losing AUC? The best setting above (pool of 140
+ 60 history features, DRFE) was run on the test years against the current model, with a margin of
0.003 AUC fixed in advance. It is exploratory, because the test years were used before.

| | Fixed 140 | Dynamic | Change (95% CI) |
|---|---|---|---|
| ROC AUC, 2003-2012 | 0.7484 | 0.7493 | +0.0008 (−0.0043 to +0.0062) |
| NDCG@5% | 0.0836 | 0.0817 | −0.0020 (−0.0146 to +0.0080) |

The verdict is **inconclusive**: the point estimate is about zero, but the interval does not rule out a
loss larger than 0.003. The pipeline therefore keeps the fixed pool. The selector also chose between 59
and 200 features for the same window depending on the seed, because validation AUC is nearly flat along
the elimination path. The honest summary is that dynamic selection makes no measurable difference
either way, and adds cost and instability. The code is in git history at commit `a05dff8`, and the
scores remain in `results/experiments/` (`cv_ni.json`, `ni_test.json`).

### Automatic feature generation: tried, did not raise ROC
Selection only chooses among existing features, so the next question was whether new features could be
built automatically inside each training window. [Protocol 5](experiments/PROTOCOL_FG.md) fixed the rules
before any run. The method is an own implementation of OpenFE's expand-and-reduce (Zhang et al., ICML
2023), because the published package does not respect time order.
* **Expand.** All 1,134 ratios and asset-scaled differences of the 28 raw items, each as a within-year
  rank. G2 adds a recursive second round on the best 25: the change from last year, the deviation from
  the firm's own 3-year mean, and ratios between them.
* **Reduce.** Boosting from the current model's margin on two inner folds that keep the 2-year gap, with
  successive halving, then ranking by split gain. The top k go to LGBM, XGB and LR.

| Setting | Gain, seed 42 | Gain, fresh seeds 43-44 (needed +0.002) |
|---|---|---|
| G1, one round, k = 10 / 25 / 50 | −0.0000 / +0.0002 / −0.0027 | k = 25: −0.0009 |
| G2, two rounds, k = 10 / 25 / 50 | −0.0021 / +0.0005 / −0.0026 | k = 25: −0.0018 |

Neither family passed, so the test years were not used. G1 raised NDCG@5% and Sens@5% on the fresh seeds,
but the rule decides on AUC and this was not tested further. Described after the result: the chosen
features change almost completely between windows and between seeds (Jaccard overlap 0.02 to 0.17), and
none was chosen in all 9 runs. With about 900 frauds, picking ratios out of over a thousand candidates is
mostly noise. LLM-proposed features (CAAFE) were not tried, because they need an external API.
The code was removed after testing; it is in git history at commit `5628dd9`, and the scores remain in
`results/experiments/` (`cv_fg*.json`).

### Raising ROC AUC towards 0.8: reached 0.748, not 0.8
Rules were fixed in advance (`experiments/PROTOCOL_AUC.md`, `experiments/PROTOCOL_TEXT.md`). The test
years had been used before, so these results are **exploratory**. The reference is the previous
ensemble: test AUC 0.7433, CV AUC 0.7723.

| Candidate on test years 2003-2012 | Fresh-seed CV AUC gain | Test AUC | Gain | 95% CI | p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|
| **Market features + recency weights + retuned trees** | **+0.0145** | **0.7484** | **+0.0051** | **[+0.0003, +0.0103]** | **0.019** | **0.17** | **Adopted, "promising, not confirmed"** |
| The same + peer deviations + Temporal Transformer × 3 | +0.0149 | 0.7482 | +0.0049 | [−0.0019, +0.0121] | 0.078 | 0.62 | Rejected |
| The same as adopted + Temporal Transformer × 3 | +0.0141 | 0.7476 | +0.0043 | [−0.0022, +0.0110] | 0.095 | 0.66 | Rejected |

* About two thirds of the validation gain did not survive on the test years (+0.0145 against +0.0051):
  those candidates had been chosen on the same validation years. The gain is positive on all 3 seeds
  (+0.0047 to +0.0054), and in 2003-2008 (+0.0072) and 2009-2012 (+0.0019). It is small and
  concentrated in the earlier years.
* Leakage audit on validation folds (`experiments/audit_auc.py`): the gain holds on frauds at firms
  with no fraud label in training (0.769 → 0.784); training on shuffled labels gives AUC 0.48; shuffling
  the market block within each year halves the gain (+0.0145 → +0.0073).
* Market features can reflect a fraud becoming public, not only fraud being committed. A fall in market
  value may follow a disclosure, so part of this signal may be early detection rather than prediction.
* Other families that failed the validation gate, by fresh-seed gain: pairwise AUC loss +0.0013, firm
  surprise features −0.0001, covariate-shift weights +0.0001, 8-year history −0.0004, abnormal accruals
  −0.0023, 6 networks −0.0024, year-aware modulation −0.0028, adding a random forest −0.0047 (the best
  of that family).
* Filing data (Loughran-McDonald 10-X summaries linked through the `farr` GVKEY-CIK table) also failed:
  industry-relative ratios +0.0014, amendment counts +0.0004, all combined −0.0001, filing timing
  −0.0006, text and tone −0.0002. Having a filing link predicts fraud on its own (flag-only AUC about
  0.57), so the features used no missingness signal, and these gains are the ones left after that
  guard. None reached the test years. The external files are not in this repository.

The code for what failed was removed after testing; it is in git history at commit `f7c0284`, and the
scores remain in `results/experiments/`.

**Why some papers report more than 0.8.** The same adopted trees and logistic regression, without the
Temporal Transformer and on seed 42, scored under weaker evaluation rules
(`experiments/diagnostic_splits.py`). These numbers are a diagnostic, not results of this project:

| Evaluation rule | AUC |
|---|---|
| **Expanding window, 2-year gap, serial-fraud masking (this project)** | **0.741** |
| The same without serial-fraud masking | 0.788 |
| Also no gap: train up to the year before the test year | 0.828 |
| Random 5-fold split by firm, 1991-2012 | 0.777 |
| Random 5-fold split of firm-years, 1991-2012 | 0.821 |

The model is the same in every row. Crossing 0.8 here needs only a weaker rule: labels from a fraud case
that spans the test year stay in training, or that case's own years fall on both sides of a random split.
The random splits average over different years than the first three rows, so compare them loosely.
A real user cannot have those labels at scoring time, because AAER labels arrive years later. So 0.8 was
not claimed from these rows.

## Explainability findings (SHAP on the deployed members, 2012-2014, real base rate)
Every member is explained directly: TreeExplainer on all LightGBM and XGBoost bags, permutation SHAP for
logistic regression, and expected gradients for the Temporal Transformer. No proxy model is used. Members are
rank-averaged, so their SHAP values cannot be added together, but `risk_percentile` is exactly the mean of the
four within-year ranks, and `top_flagged_explained.csv` lists each rank next to the drivers.

### Tree members (LightGBM + XGBoost)
Top drivers: firm size (`log_sale`, `log_at` and their ranks), security issuance, gross PPE / assets,
external financing need, investments / assets, depreciation / assets, receivables / sales. Market-value
volatility is the strongest new market feature, 11th overall.
**Firm size still dominates** even without raw dollar items. Dropping `log_at`/`log_sale` from the
feature set would make the model size-neutral; it was kept because the paper's own features
carry size. See `results/explain/` (bar, beeswarm, dependence plots, waterfalls,
`top_flagged_explained.csv` with `risk_percentile` + calibrated `risk_score`).
Caveat: SHAP explains what the model uses, not why a firm committed fraud; labels are SEC-enforced
cases only (undetected fraud is labelled 0, and 2011+ labels are still being discovered).

### Temporal Transformer
**Method.** Expected gradients (Erion et al. 2021), the method behind `shap.GradientExplainer`, on the same
3,000 firm-years, with 50 complete-history training firm-years as the reference. The output explained is the
logit, in the same log-odds units as the tree members. Only reported values get attribution: a firm-year's
missing-value flags and absent years stay fixed, so the method does not measure what being missing contributes.

**Findings.** See `temporal_shap_*` and `waterfall_tt_*` in `results/explain/`.
* Top features: rank of sales, the loss flag, ranks of depreciation / assets, Altman Z, retained earnings /
  assets and soft assets. Five of its top 10 are in the tree members' top 10 (ranks of sales, assets, PPE,
  issuance and depreciation). The loss flag, Altman Z, retained earnings and soft assets are not.
* Share of attribution by year, t-4 to t: 9%, 12%, 17%, 23%, 38%. Part of this slope comes from older years
  existing for fewer firm-years (t-4 for 58%), but it remains among firm-years that have the year: mean size
  0.157, 0.191, 0.227, 0.271, 0.383.

**Audit.** [Protocol 4](experiments/PROTOCOL_SHAP.md) set the checks and limits before any attribution was computed.
Seven of eight passed, so the audit as a whole did not pass.

| | Check | Result | Limit | Verdict |
|---|---|---|---|---|
| A | Known answer on a linear network; missing cells get nothing | error 4.8e-07; exactly 0 | below 1e-4; exactly 0 | Pass |
| B | Same output as the deployed network; same firm ranking | gap 9.5e-07; Spearman 0.997 | below 1e-4; at least 0.99 | Pass |
| C | Attributions add up to the output (error in output SDs) | median 0.004; 99th percentile 0.020 | 0.01; 0.05 | Pass |
| D | Agrees with permutation and exact Shapley on 300 rows | global 0.986; within firm-year 0.974; year-share gap 0.037 | 0.80; 0.80; 0.05 | Pass |
| E | Stable across reference samples; across the 3 networks | 0.984; 0.806 | 0.95; 0.70 | Pass |
| F | Resetting a row's top 20 cells lowers the logit, against 20 random cells | 0.962 vs 0.018 (53.6 times); CI of the difference [0.919, 0.970] | 2 times; above 0 | Pass |
| **G** | **Global importance against re-initialised weights; shuffled labels** | **0.425; 0.179** | **0.30; 0.50** | **Fail** |
| H | Same result on a repeat run | difference 0.0 | below 1e-3 | Pass |

**What the failure means.** An untrained network shares part of the real model's global feature ordering.
A follow-up on 10 random networks, chosen after seeing the failure so it cannot overturn it, gave the following
(`experiments/audit_temporal_shap_g.py`):
* Global ranking: mean Spearman 0.35 (range 0.22 to 0.51). How much each input varies does not explain this.
  Why it happens is not established.
* Within a firm-year: median correlation with the real attributions of -0.03. Resetting the 20 cells a random
  network picks raises the real model's logit by 0.09 on average, while the real attributions' 20 lower it by 0.96.

So explanations of an individual firm-year depend on the trained model and change its output, and they are the
ones to rely on. The Transformer's global feature ranking is stable and agrees with a second method, but it
is indicative only. The sample has 5 frauds, so these checks show how faithfully the model is explained and
say nothing about why any firm committed fraud.

**Effect on the model.** None. The rerun's model scores, metrics and tree SHAP values are identical to before.
It exposed one existing problem, now fixed: the unseeded logistic regression SHAP column changed on every run
(by up to 0.005, Spearman 0.999), and it is now seeded. That column is not used by any plot or score.

## Scoring new firms
`score_new_data.py` ranks the input firms against the **full market cross-section of the same
fiscal year** taken from `--reference` (the training CSV by default), because 73 of the features
and the final score are within-year ranks: scoring 10 firms on their own would rank them only
against each other. Include each firm's prior-year row so lag features exist. The Temporal
Transformer reads the firm's last 5 fiscal years and the market features read up to 4 earlier ones:
years missing from the input come from `--reference`, and a firm in neither file is scored on the
shorter history it has. Only the `--year`
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

## Fresh-period test (fiscal 2015-2018, SEC XBRL data, new labels)

Protocol: `experiments/PROTOCOL_FRESH.md`, written before any fresh score existed. Code: `experiments/fresh/`.
The saved model (trained on fiscal 1991-2010) was scored once, unchanged, on data that is not in the paper's
file: financials from the SEC's XBRL API as first reported, and fraud labels written by hand from the AAERs
released 2017-2024 (`experiments/fresh/aaer_labels_fresh.csv`).

| Test | Firm-years | Frauds | AUC | 95% CI |
|---|---|---|---|---|
| **Fiscal 2015-2017, mean of yearly AUCs (primary)** | 12,474 | 55 | **0.776** | [0.721, 0.828] |
| Fiscal 2015-2018 pooled | 16,478 | 67 | 0.769 | [0.711, 0.820] |
| Only cases the SEC calls fraud, pooled | 16,449 | 38 | 0.773 | [0.689, 0.836] |
| Only filer-years with a price | 14,086 | 60 | 0.766 | [0.712, 0.815] |
| Only total assets of at least $10m | 12,310 | 64 | 0.747 | [0.697, 0.799] |
| Fiscal 2019-2020, labels incomplete, pooled | 7,974 | 8 | 0.779 | [0.637, 0.939] |

Yearly AUC: 0.791 (2015), 0.805 (2016), 0.733 (2017), 0.727 (2018, 12 frauds, below the 15-fraud rule so not
in the mean). Verdict under the protocol: **Holds** (bar: 0.725, the lower end of the 2003-2012 interval).

* Bridge check on fiscal 2012-2014, same 8,874 firm-years and 22 frauds: AUC 0.701 on Compustat inputs,
  0.700 on XBRL inputs. The change of data source costs nothing measurable.
* Top-of-list results are weaker in the later years: Sens@5% is 0.32, 0.11, 0.06 and 0.08 for 2015-2018.
* 9 of the 84 labelled firm-years for 2015-2020 are not in the universe (banks and lenders without a
  classified balance sheet, or no matching 10-K row) and are not scored.
* Limits: the price is public float / shares, cost of goods sold is missing for about a third of filers,
  the labels come from one reader with no second coder, and releases from 2025-2026 are not covered.

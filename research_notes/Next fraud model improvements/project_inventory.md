# Project inventory: what has been tried, what worked, what data exists (local project, read-only review)

All sources are local files under `D:/Final Year Project/`; links below are relative to that folder. Numbers were copied from the files named. "CV" = validation AUC on rolling-origin folds that validate on fiscal 1999, 2000 and 2001 (pre-2003 data only). "Fresh-seed gain" = gain on seeds 43 and 44 against the reference, which is the gate the later protocols used. "Test" = mean of yearly AUCs over test years 2003-2012, 3 seeds, unless stated otherwise.

## Q1. Every experiment family tested so far, with its result and where it is documented

### Takeaway
About 60 settings in 9 protocol families have been tested. Only three changes were ever adopted: the Temporal Transformer member (+0.0084 test AUC), and the combination of 7 market-value rank features + recency weights (half-life 3) + trees retuned for the ensemble (+0.0051 test AUC). Both are labelled "promising, not confirmed", because neither survives a Holm correction. Everything else failed the +0.002 fresh-seed CV gate or lost on the test years: feature selection (10 families plus DRFE and a non-inferiority test), automatic feature generation, feature-token deep models, PU loss, RF/ET, peer, accrual and surprise features, pairwise AUC loss, covariate-shift weights, and filing text/industry/timing/amendments.

### Cited Findings

**Protocol 1: DRFE, sequence/deep models, TabM, iFPU (reference = LGBM+XGB+LR on 133 features; ref CV 0.7597, ref test 0.7349)**
- Rules: a candidate needs at least +0.002 mean CV AUC over the reference. Sequence models were also scored on the test years as standalone models whatever their CV result. Adoption needed P(gain<=0) <= 0.05 on some metric, no metric with P >= 0.90, gain positive on all 3 seeds and in both 2003-2008 and 2009-2012. Holm-adjusted p is reported, and a candidate that fails Holm is labelled "promising, not confirmed" — [experiments/PROTOCOL.md](experiments/PROTOCOL.md)
- Deep epoch amendment: a bug stopped tuning at epoch 20. FT-Transformer and Mambular peaked at epoch 20, so they were re-run with epochs 30/40/50/60. Every longer run scored lower (FTT 0.689/0.669/0.623/0.631; Mambular 0.683/0.653/0.649/0.610 against 0.6996/0.6993 at epoch 20), so epoch 20 was frozen — [experiments/PROTOCOL.md](experiments/PROTOCOL.md); [results/experiments/log_deep_cv2.txt](results/experiments/log_deep_cv2.txt)
- **+ Temporal Transformer** (each firm's last 5 fiscal years as the sequence): CV gain +0.0126 (0.7723 vs 0.7597). Test AUC 0.7433 vs 0.7349, gain **+0.0084**, 95% CI [+0.0036, +0.0135], p = 0.003, Holm p = 0.108. Positive on all seeds (+0.0076 to +0.0090) and in both periods (+0.0098 in 2003-08, +0.0062 in 2009-12). NDCG@5% +0.003 (p 0.25) and Sens@5% +0.007 (p 0.23) are not significant. **ADOPTED, "promising, not confirmed"** — [README.md](README.md); [results/experiments/test_comparison.csv](results/experiments/test_comparison.csv); [results/experiments/test_means.csv](results/experiments/test_means.csv)
- **+ Temporal Mamba**: CV gain +0.0041. Test 0.7383, gain +0.0034, CI [−0.0003, +0.0074], p = 0.036, Holm 1.0. It met the adoption rule but is weaker than the TT, so it was not added — [README.md](README.md); [results/experiments/test_verdicts.json](results/experiments/test_verdicts.json)
- **+ FT-Transformer**: CV −0.0036. Test 0.7281 (−0.0069, p(gain<=0) = 1.0). Rejected. Alone: CV 0.6896, test 0.6756 — [results/experiments/cv_deep.json](results/experiments/cv_deep.json); [results/experiments/test_means.csv](results/experiments/test_means.csv)
- **+ Mambular**: CV −0.0011. Test 0.7289 (−0.0061). Rejected. Alone: CV 0.7030, test 0.6707 — same files
- **+ TabM** (Gorishniy et al., ICLR 2025): CV −0.0026. Test 0.7262 (−0.0087). Rejected. Alone: CV 0.7153, test 0.6740 — same files
- TT alone: CV 0.7574, test 0.7195. Temporal Mamba alone: CV 0.7466, test 0.7205 — [results/experiments/test_means.csv](results/experiments/test_means.csv)
- Frozen deep configs: TT d=32, 2 layers, dropout 0.2, lr 1e-3, wd 1e-5, 1:10 sampling ratio, 15 epochs, 3 networks. Temporal Mamba d=64, 1 layer, d_state 8, 10 epochs. FTT d=32, 2 layers. Mambular d=16. TabM d=16, width 256, 2 layers, 15 epochs — [results/experiments/cv_deep.json](results/experiments/cv_deep.json); [src/deep_models.py](src/deep_models.py)
- **Dynamic RFE** (tau x rule grid, re-run per window): best CV is tau 0.02 "best" (111 features, 0.7599) and tau 0.02 one_se (94 features, 0.7599), +0.0003 over the reference. The tau 0.02 one_se setting was scored on the test years anyway: 0.7325, gain −0.0024 (p 0.887), with harm on Sens@5% (p 0.92). Rejected. Larger tau dropped CV to 0.737-0.755 — [results/experiments/cv_drfe.json](results/experiments/cv_drfe.json); [results/experiments/test_comparison.csv](results/experiments/test_comparison.csv)
- **Focal PU XGBoost (iFPU risk, Zavitsanos & Paliouras, arXiv 2605.14467)**, grid pi in {0.01, 0.02, 0.05, 0.1} x gamma in {1, 3} x mode {alone, replace XGB, add}: best is pi 0.01, gamma 3, replace, CV 0.7601 vs 0.7596 (+0.0005). Test 0.7310, gain −0.0039, CI [−0.0067, −0.0012], p(gain<=0) = 0.996, so it measurably hurt AUC. "Alone" modes scored 0.696-0.745 CV. Rejected — [results/experiments/cv_pu.json](results/experiments/cv_pu.json); [results/experiments/test_comparison.csv](results/experiments/test_comparison.csv); code in git commit `ae9b5ed` (`src/pu_learning.py`)
- Leakage audit for the TT/TMamba: history entries are always the same firm in consecutive earlier years (0 violations over 141,463 rows, mean history length 3.57). Training on labels shuffled within each year gives AUC 0.529 (TT) and 0.547 (TMamba). On first-time frauds (231 of 240 validation frauds) the ensemble goes 0.757 → 0.769 — [results/experiments/audit_temporal.json](results/experiments/audit_temporal.json); [README.md](README.md)

**Protocol 2: dynamic feature selection, 10 families (reference = LGBM+XGB+LR+TT, CV 0.7723, test 0.7433)**
- Fresh-seed CV gains of each family's best setting: F9 pool of 133 + 60 firm-history features then DRFE (tau 0.02, best) +0.0008; F6 DRFE for LR only (0.05, one_se) −0.0002; F7 soft importance-sampled features per bag (alpha 0.5) −0.0018; F10 per-firm-year gates in the TT (sigmoid, 68 inputs) −0.0025; F5 mRMR (keep 100) −0.0026; F8 DRFE-subset LGBM as an extra member (0.05) −0.0041; F4 adversarial drift filter (drop 10%) −0.0047; F3 out-of-time permutation importance −0.0067; F1 stability selection (freq 0.6) −0.0207; F2 Boruta shadow features (q 0.75) −0.0311. Nothing passed, so the **test years were not used** — [README.md](README.md); [results/experiments/cv_fs.json](results/experiments/cv_fs.json); [experiments/PROTOCOL_FS.md](experiments/PROTOCOL_FS.md)
- The strictest screen settings collapsed (F1_0.95 −0.0971 with about 3 features; F2_1.0 −0.0868 with about 2 features) — [results/experiments/log_fs_screen.txt](results/experiments/log_fs_screen.txt)
- Code removed; it is in git history at commit `255e803` (`experiments/run_fs.py`, `src/dynamic_selection.py`) — [README.md](README.md)

**Protocol 3: raising ROC AUC with the existing data (reference CV 0.7723 on seeds 43/44, 0.7743 on seed 42; test 0.7433)** — [experiments/PROTOCOL_AUC.md](experiments/PROTOCOL_AUC.md); [results/experiments/cv_auc_screen.json](results/experiments/cv_auc_screen.json); [results/experiments/cv_auc.json](results/experiments/cv_auc.json)
- A1, 7 market features (within-year ranks of market-value growth, its lag, 3-year growth, volatility over up to 4 years, share-count growth, market growth minus asset growth, market growth minus sales growth, all from prcc_f x csho): screen +0.0055, fresh +0.0057, **pass**
- A2, peer deviations (20 features vs per-year k-means peers, k = 20, standing in for the missing industry codes): screen +0.0023, fresh +0.0020, **pass** (only inside a combination that was later rejected)
- A3, abnormal accruals (modified-Jones residuals within peer groups, 4 features): −0.0023 / −0.0023, fail
- A4, recency weights: half-life 3 years gives screen +0.0057, fresh +0.0059, **pass**; half-life 6 gives screen +0.0024
- A5, extra learners: random forest −0.0007 screen / −0.0047 fresh; extra trees −0.0062; RF+ET −0.0062; fail
- A6, 6 TT networks instead of 3: −0.0018 / −0.0024, fail
- A7, unequal blend weights: TT x 3 screen +0.0068, fresh +0.0042 (pass alone); TT x 2 +0.0051; no LR −0.0046; TT x 2 without LR +0.0013; + Temporal Mamba −0.0018
- A8, 8-year TT history: +0.0014 / −0.0004, fail
- A9, year-aware input modulation in the TT (after arXiv 2512.03678): −0.0031 / −0.0028, fail
- A10, surprise features (actual minus predicted from the firm's prior year, 20 features): −0.0008 / −0.0001, fail
- A11, retune trees for the ensemble (40 trials each): LGBM +0.0032, XGB +0.0017, both +0.0047 screen / +0.0046 fresh, **pass**
- A12, pairwise AUC loss in the TT: −0.0012 screen / +0.0013 fresh, fail
- A13, covariate-shift weights (train-vs-evaluation-year classifier, clip 5 or 10): +0.0002 / +0.0001, fail
- Combinations, fresh-seed: A1+A4 +0.0116; **A1+A4+A11 +0.0145 (CV 0.7868)**; A1+A4+A11+A7tt3 +0.0141; A1+A2+A4+A11+A7tt3 +0.0149 (CV 0.7873) — [results/experiments/cv_auc.json](results/experiments/cv_auc.json)
- Test (2003-2012, against 0.7433): **A1+A4+A11 → 0.7484, +0.0051, CI [+0.0003, +0.0103], p 0.019, Holm 0.17, ADOPTED "promising, not confirmed"**. Per seed +0.0047 to +0.0054; 2003-08 +0.0072, 2009-12 +0.0019. NDCG@5% +0.0015 (p 0.41), Sens@5% −0.0014 (p 0.55). The version with peers and TT x 3 scored 0.7482 (+0.0049, p 0.078, Holm 0.62) and the one with TT x 3 only scored 0.7476 (+0.0043, p 0.095, Holm 0.66); both rejected — [results/experiments/auc_test_comparison.csv](results/experiments/auc_test_comparison.csv); [results/experiments/auc_test_verdicts.json](results/experiments/auc_test_verdicts.json)
- About two thirds of the validation gain did not reach the test years (+0.0145 → +0.0051) — [README.md](README.md)
- Audit: on first-time frauds (231 of 240) CV goes 0.769 → 0.784. A trees-only placebo gives 0.478. Shuffling the market block within each year halves the gain (0.0145 → 0.0073). The README warns that market features may partly reflect a fraud becoming public — [results/experiments/audit_auc.json](results/experiments/audit_auc.json); [README.md](README.md)
- Retuned tree parameters frozen in `src/models.py`. LGBM: 289 trees, lr 0.0371, 14 leaves, min_child 38, colsample 0.79, subsample 0.999, lambda 0.148, ratio 20. XGB: 377 trees, lr 0.0239, depth 6, min_child_weight 2.35, colsample 0.61, subsample 0.64, lambda 0.405, ratio 20 — [results/experiments/auc_retune.json](results/experiments/auc_retune.json); [src/models.py](src/models.py)
- Code removed; it is in git history at `f7c0284` (`experiments/auc_features.py`, `run_auc.py`, `filing_features.py`, `run_text.py`) — [README.md](README.md)

**Protocol 3b: filing data (Loughran-McDonald 10-X summaries linked through farr GVKEY-CIK)** — [experiments/PROTOCOL_TEXT.md](experiments/PROTOCOL_TEXT.md); [results/experiments/cv_text.json](results/experiments/cv_text.json); [results/experiments/cv_text_screen.json](results/experiments/cv_text_screen.json)
- The gate needed +0.002 on all rows, +0.002 on linked rows, and a placebo gain of at most +0.001. Unlinked rows got values copied from a random linked row of the same year, so there is no missingness flag
- Linkage alone is a fraud signal: 93-96% of fraud firm-years link against 78-81% of the others, and a flag-only AUC is 0.567-0.584 across the validation years
- T1 text size/vocabulary/tone: fresh −0.0002 (linked −0.0007, placebo +0.0026). T2 industry ratios minus the FF48 industry-year median plus the industry training fraud rate: screen −0.0115 (overfit). T2n, no rate: fresh +0.0014 (linked +0.0023, placebo −0.0029). T2p, lagged rate: screen +0.0006. T3 filing timing: −0.0006. T4 amendments in the prior 3 years: +0.0004. T1+T2p+T3+T4: −0.0001. **None passed, so the test years were not used**

**Protocol 4: SHAP audit of the TT (not a model change)**
- 7 of 8 pre-set checks passed. Check G failed: global importance against re-initialised weights had Spearman 0.425 (limit 0.30), and against shuffled labels 0.179. So the per-firm-year explanations are reliable and the global ranking is indicative only. Attribution share by year, t-4 to t: 9/12/17/23/38% — [README.md](README.md); [experiments/PROTOCOL_SHAP.md](experiments/PROTOCOL_SHAP.md); [results/experiments/audit_temporal_shap.json](results/experiments/audit_temporal_shap.json)

**Protocol 5: automatic feature generation (OpenFE-style expand-and-reduce; reference CV 0.7868)** — [experiments/PROTOCOL_FG.md](experiments/PROTOCOL_FG.md); [results/experiments/cv_fg.json](results/experiments/cv_fg.json)
- G1 builds 1,134 ratios and asset-scaled differences of the 28 raw items, then keeps the top k. Seed-42 gains: k=10 −0.0000, k=25 +0.0002, k=50 −0.0027. Fresh k=25: −0.0009 (NDCG@5% 0.215 vs 0.207, Sens@5% 0.299 vs 0.284)
- G2 adds a recursive round (lag change, deviation from the firm's 3-year mean, ratios). Seed-42 gains: k=10 −0.0021, k=25 +0.0005, k=50 −0.0026. Fresh k=25: −0.0018
- Selected features were unstable: Jaccard 0.17 / 0.06 (G1, across seeds / across windows) and 0.08 / 0.02 (G2). No feature was chosen in all 9 runs. Test years not used. Code at `5628dd9` — [results/experiments/fg_selection_stability.json](results/experiments/fg_selection_stability.json)
- CAAFE (LLM-proposed features) was explicitly not tried, because it needs an external API — [experiments/PROTOCOL_FG.md](experiments/PROTOCOL_FG.md)

**Protocol 6: non-inferiority of dynamic selection (F9 on a 140 + 60 history pool, DRFE tau 0.02)** — [experiments/PROTOCOL_NI.md](experiments/PROTOCOL_NI.md); [results/experiments/ni_test.json](results/experiments/ni_test.json)
- CV 0.7843 vs 0.7868 (−0.0025). Test AUC 0.7493 vs 0.7484, +0.0008, CI [−0.0043, +0.0062]. NDCG@5% −0.0020, Sens@5% +0.0003, Prec@1% +0.0009. Verdict **inconclusive** with a margin of 0.003, so the fixed pool is kept. Features chosen ranged from 59 to 200 per window and seed (mean 134). Code at `a05dff8`

**Protocol 7: fresh period (fiscal 2015-2018, SEC XBRL inputs, hand-written AAER labels for 2017-2024 releases), saved model frozen** — [experiments/PROTOCOL_FRESH.md](experiments/PROTOCOL_FRESH.md); [results/experiments/fresh_test.json](results/experiments/fresh_test.json)
- Primary, mean of yearly AUCs over 2015-2017 (12,474 firm-years, 55 frauds): **0.776**, CI [0.721, 0.828]. Verdict "Holds" (bar 0.725). Pooled 2015-2018 (16,478 rows, 67 frauds): 0.769. Yearly: 0.791, 0.805, 0.733, and 0.727 for 2018 (12 frauds)
- Sensitivities: tier A only 0.773 pooled; filer-years with a price 0.766; assets of at least $10m 0.747; quarter-only cases counted as fraud 0.788. Members alone, mean yearly: LGBM 0.760, XGB 0.774, LR 0.738, TT 0.728. Sens@5% by year: 0.32, 0.11, 0.06, 0.08
- Bridge, fiscal 2012-2014 on 8,874 common rows with 22 frauds: Compustat inputs 0.701, XBRL inputs 0.700. Rank correlation of the two score sets 0.844 — [results/experiments/fresh_bridge.json](results/experiments/fresh_bridge.json)

**Protocol 8: completed labels for fiscal 2011-2014** — [experiments/PROTOCOL_LABELS.md](experiments/PROTOCOL_LABELS.md); [README.md](README.md)
- 19 firm-years were relabelled as fraud. Frauds per year: 2011 21→23, 2012 19→20, 2013 11→16, 2014 4→15. Proposed vs baseline: re-scored 2003-2012 0.750 vs 0.708; re-scored 2003-2014 0.760 vs 0.716; **retrained 2003-2014 0.761 vs 0.717**, gain +0.044 [+0.025, +0.064]. Yearly, retrained: 2011 0.772, 2012 0.781, 2013 0.771, 2014 0.856
- In the retrained run's log, the proposed model has 2003-2014 AUC 0.7609, CI [0.742, 0.782]. "Ensemble without firm history" scores 0.7530 and "previous ensemble" 0.7547 — [results_labels/log.txt](results_labels/log.txt)

**Diagnostic, not a candidate: evaluation rules (adopted trees+LR without TT, seed 42)** — [results/experiments/diagnostic_splits.json](results/experiments/diagnostic_splits.json)
- Project rules 0.741; no serial-fraud masking 0.788; also no gap 0.828; random firm 5-fold 0.777; random firm-year 5-fold 0.821

### Inferences
- Measured gains are tiny. Even the adopted changes sit around +0.005 to +0.008 test AUC, and validation gains shrink by about two thirds on the test years. Any new idea should expect a CV gain of at least 0.005 to have a chance of a significant test gain.
- The ideas already ruled out include: feature selection of any kind, OpenFE-style ratio expansion, feature-token deep tabular models (FTT/Mambular/TabM), PU/focal loss, RF/ET members, larger or longer or year-modulated TT variants, pairwise AUC loss, covariate-shift weights, surprise features, abnormal accruals, and LM text/tone, timing, amendments and FF48 industry deviations.
- What helped is new information or better use of time: firm history (TT), market-value movement, recency weighting. This points towards new data sources, or modelling label delay and time, rather than new learners on the same 28 items.
- Peer deviations (A2) passed the gate alone (+0.0020) but were tested on the test years only in a combination that failed. A2 on its own on top of the adopted model was never tested on the test years.
- G1 feature generation raised NDCG@5% and Sens@5% on fresh seeds, but the protocol decides on AUC. A top-of-list objective is a separate, untested question.

### Gaps
- `cv_fs_screen.json` holds every F-family setting. Only the best per family was transcribed here; the full grid is in that file and in `log_fs_screen.txt`.
- The test-year numbers for 2013-2014 under the paper's labels are excluded (fewer than 15 frauds), so no protocol-level result exists for them except Protocol 8.
- `labels_rescore.json` and `labels_retrained.json` were not opened field by field. The figures above come from README.md and `results_labels/log.txt`.

## Q2. Adoption rule, evaluation protocol, and the current final model with its metrics

### Takeaway
Evaluation is the paper's own: expanding window, 2-year gap, serial-fraud masking, test years 2003-2012 (2013-2014 excluded for having fewer than 15 frauds). A candidate must first gain at least +0.002 CV AUC on fresh seeds 43/44, then show a significant test AUC gain (p <= 0.05) that is positive on all 3 seeds and in both sub-periods, with no metric harmed (p >= 0.90). The final model is LGBM + XGB + LR on 140 scale-free features + Temporal Transformer, rank-averaged. It scores 0.748 test AUC (2003-2012), 0.776 on fresh 2015-2017 data, and 0.761 on completed labels for 2003-2014.

### Cited Findings
- Protocol: for test year T, train on 1991..T-2 (2-year gap). Training rows of an AAER case that also appears in the test year are set to 0 (serial-fraud masking). Winsorizer refit per window. Hyper-parameters tuned only on pre-2003 rolling-origin CV. Test rows are never subsampled. Years with fewer than 15 frauds are excluded from means — [run_pipeline.py](run_pipeline.py) docstring
- CV folds validate on 1999, 2000 and 2001, with `make_folds(last_train_years=(1997,1998,1999), gap=1)`. The README warns that only 3 validation years with about 300 frauds are used, so differences of about 0.005 are within noise — [src/models.py](src/models.py); [README.md](README.md); [experiments/common.py](experiments/common.py)
- Gate: at least +0.002 mean CV AUC on fresh seeds 43 and 44, after a seed-42 screen. At most 3 settings advance (2 in Protocol 5). Test-stage adoption: P(AUC gain <= 0) <= 0.05; no metric among AUC, NDCG@5% and Sens@5% with P >= 0.90; AUC gain positive on all 3 seeds and in 2003-2008 and 2009-2012. Paired bootstrap with 1000 resamples, frauds clustered by AAER case and non-frauds by firm. Holm-adjusted p is reported — [experiments/PROTOCOL_AUC.md](experiments/PROTOCOL_AUC.md); [experiments/PROTOCOL_FS.md](experiments/PROTOCOL_FS.md); [experiments/PROTOCOL.md](experiments/PROTOCOL.md)
- Every protocol after Protocol 1 states that the 2003-2012 test years have been reused, so its results are "exploratory" — [experiments/PROTOCOL_AUC.md](experiments/PROTOCOL_AUC.md); [experiments/PROTOCOL_FG.md](experiments/PROTOCOL_FG.md)
- Final model: balanced-bagged LightGBM and XGBoost (10 bags, 1:20 sampling, recency weights 0.5^((newest year − year)/3)), logistic regression, and a Temporal Transformer over the firm's last 5 fiscal years of the `rank_only` group (68 inputs), 3 networks. Members are rank-averaged within each year. The deployable model trains on fiscal 1991-2010 — [README.md](README.md); [src/models.py](src/models.py); [src/deep_models.py](src/deep_models.py); [run_pipeline.py](run_pipeline.py)
- Test 2003-2012, seed 42: proposed AUC 0.748, NDCG@5% 0.081, Sens@5% 0.151, Prec@1% 1.9%, Lift@1% 2.7. Baseline RUSBoost on the raw 28 items: 0.705 / 0.066 / 0.098 / 1.8% / 3.1. RUSBoost on the enhanced features 0.732; LGBM alone 0.730; TT alone 0.721; previous ensemble 0.743; ensemble without TT 0.740 — [README.md](README.md)
- Gain over the baseline: AUC +0.044 [+0.025, +0.062], p 0.000; NDCG@5% +0.015 [−0.011, +0.042], p 0.118 (not significant); Sens@5% +0.053 [+0.013, +0.093], p 0.003 — [README.md](README.md)
- The latest gain improves the ranking as a whole, not its top. Over 3 seeds, Prec@1% falls from 2.1% to 1.9% against the previous ensemble — [README.md](README.md)
- CV progression: raw 28 → 0.690; enhanced_rank ensemble → 0.760; + TT → 0.772; + market, recency weights and retuned trees → 0.787 — [README.md](README.md)
- Base rate is about 0.6% of firm-years. The calibrated top decile has observed rate 2.2% and predicted 2.2%; calibration saturates near 2-3% — [README.md](README.md)
- Fresh period 0.776 (2015-2017) and completed labels 0.761 (2003-2014): see Q1, Protocols 7 and 8

### Inferences
- With about 20-90 frauds per test year, the test-year CI half-width for a gain is about ±0.005. A new idea therefore needs roughly +0.006 test AUC to be significant, more than either adopted change achieved alone, except the TT.
- Because the test years are spent, the fresh XBRL period (2015-2018) is the only partly clean confirmation set. It has been used once, for the frozen model.

### Gaps
- The README states selection uses "~300 frauds" in validation. Counted from the data file: 73 + 86 + 81 = 240 fraud firm-years in 1999-2001, matching "240 frauds" in `audit_auc.json`. The 300 figure is not reproduced.

## Q3. What data is available on disk, and what is not

### Takeaway
The core file is the paper's 146,045 firm-years (fiscal 1990-2014, 18,444 gvkeys, 964 fraud firm-years) with 28 raw Compustat items and 14 ratios: no SIC, no CIK, no cash flow, no SG&A, no returns. On disk (git-ignored) there are also: the farr GVKEY-CIK link table; the full Loughran-McDonald 10-X summaries 1993-2025, which include SIC and FF industry for every filing; SEC XBRL facts for about 14,339 filers (fiscal 2010-2021, mapped to the 28 items plus 15 extra tags such as SG&A, R&D, gross profit and operating income); 697 AAER texts; and the paper's own AAER_firm_year file. There is no CRSP, Audit Analytics, I/B/E/S, governance/board, or full Compustat.

### Cited Findings
- `data/data_FraudDetection_JAR2020.csv`: 146,045 rows x 46 columns. Columns: `fyear, gvkey, p_aaer, misstate`; raw 28: `act, ap, at, ceq, che, cogs, csho, dlc, dltis, dltt, dp, ib, invt, ivao, ivst, lct, lt, ni, ppegt, pstk, re, rect, sale, sstk, txp, txt, xint, prcc_f`; 14 ratios: `dch_wc, ch_rsst, dch_rec, dch_inv, soft_assets, ch_cs, ch_cm, ch_roa, issue, bm, dpi, reoa, EBIT, ch_fcf` — file header; [src/features.py](src/features.py)
- Counted from the file: fiscal 1990-2014, 18,444 firms, 964 fraud firm-years, 412 distinct `p_aaer` cases. Frauds per year rise from 15 (1990) to a peak of 86 (2000), then fall: 69 (2003), 58, 45, 33, 30, 26 (2008), 31, 26, 21, 19, 11 (2013), 4 (2014). There are 5,400-6,800 firm-years per year — computed with pandas from the CSV (this review)
- `data/data_completed_labels.csv`: same rows, 983 frauds. 2009-2014: 31, 26, 23, 20, 16, 15 — computed from the CSV; label source [experiments/labels/labels_completed.csv](experiments/labels/labels_completed.csv)
- 140 features: 14 paper ratios; engineered levels (log size, ROA, margins, liquidity, leverage, 14 items / assets, fin_need, loss flag); growth gaps; changes; partial Beneish components (SGAI fixed at 1, gross PPE, working-capital accruals) and Altman Z; within-year percentile ranks of all of these (`enhanced_rank`, 133); plus 7 market-value rank features — [src/features.py](src/features.py); [README.md](README.md)
- Explicit absences in the paper data: industry codes are missing ("standing in for the industry codes this dataset lacks", A2 k-means peers) — git `f7c0284:experiments/auc_features.py`. SGAI is set neutral because SG&A is missing; accruals come from working capital because operating cash flow is absent — [src/features.py](src/features.py); [README.md](README.md)
- `data/external/farr/gvkey_ciks.csv`: 49,089 rows, columns `gvkey, iid, cik, first_date, last_date` (from the R package farr 1.1.1, tarball also present) — file header
- `data/external/lm10x/Loughran-McDonald_10X_Summaries_1993-2025.csv` (197 MB, 1,249,506 filings, filed 1993-08-13 to 2025-12-31). Columns: CIK, FILING_DATE, ACC_NUM, CPR, FORM_TYPE, CoName, **SIC, FFInd**, N_Words, N_Unique_Words, N_Negative, N_Positive, N_Uncertainty, N_Litigious, N_StrongModal, N_WeakModal, N_Constraining, N_Complexity, N_Negation, GrossFileSize, NetFileSize, NonTextDocTypeChars, HTMLChars, XBRLChars, XMLChars, N_Exhibits. Forms include 10-Q (706,255), 10-K (232,660), 10QSB, 10-K-A (47,523), 10-Q-A, 10KSB, 10-K405. SIC has no missing values — file header and pandas count (this review)
- `data/external/fresh/xbrl/`: `universe.csv` has 14,339 CIKs. `facts/` holds 14,339 gz files (68 MB), each with first-reported 10-K values for about 103 us-gaap tags mapped to 27 Compustat items (no price tag; price = public float / shares). `facts_extra/` holds 14,339 files (43 MB) with 15 more tags: AmortizationOfIntangibleAssets, CommonStockSharesIssued, CostsAndExpenses, DepreciationNonproduction, GeneralAndAdministrativeExpense, GrossProfit, OperatingExpenses, OperatingIncomeLoss, OtherDepreciationAndAmortization, ResearchAndDevelopmentExpense, SellingAndMarketingExpense, SellingGeneralAndAdministrativeExpense, and weighted-average diluted and basic shares. Fiscal years 2010-2021 were requested; the download took about 50-54 minutes at the SEC rate limit — [experiments/fresh/xbrl_download.py](experiments/fresh/xbrl_download.py); sampled file contents; [data/external/fresh/xbrl/download.log](data/external/fresh/xbrl/download.log)
- Not downloaded among the XBRL tags: operating cash flow (e.g. NetCashProvidedByOperatingActivities) is not in TAGS or in the extra tags sampled — [experiments/fresh/xbrl_download.py](experiments/fresh/xbrl_download.py) and sampled files (this review; based on a 1-in-200 file sample)
- `data/external/fresh/xbrl_firm_years.csv`: 63,074 firm-years, 9,318 CIKs, fyear 2008-2026. Columns: fyear, gvkey, cik, name, fye, filed, p_aaer, misstate, the 28 items, ppent and the 14 ratios. Labels are applied later from `experiments/fresh/aaer_labels_fresh.csv` — file header and pandas count
- `data/external/fresh/hf_aaer/`: a subset of the Hugging Face "SEC-RPT-Panel v2.0.0". On disk: `aaer_manifest.csv` (2,594 AAERs, release years 2004-2024), `fraud_labels.csv` (504 rows, only 24 firms, 20 positives; a related-party-transaction panel), and `txt/` with 697 AAER texts (AAER-3841 to AAER-4554). The panel's other tables named in its README (nlp_scores, xbrl_financials, master_panel, company_universe with SIC, proxy_manifest) are **not** on disk — [data/external/fresh/hf_aaer/README.md](data/external/fresh/hf_aaer/README.md); directory listing
- `data/external/fresh/jarfraud/`: `AAER_firm_year.csv` from the paper's GitHub (1,746 rows: P_AAER, CIK, YEARA, UNDERSTATEMENT; AAERs up to 4012, Dec 2018) and `identifiers.csv` (fyear, gvkey only) — file headers; [experiments/PROTOCOL_LABELS.md](experiments/PROTOCOL_LABELS.md)
- Hand labels: `experiments/fresh/aaer_labels_fresh.csv` (57 data rows, fiscal 2015-2020), `experiments/labels/aaer_labels_pre2015.csv` (17 rows), `labels_added.csv` (41 rows) — line counts
- The README says "the external files are not in this repository": `data/external/` is git-ignored but present locally — [README.md](README.md); [.gitignore](.gitignore)
- Cached experiment scores: `results/experiments/cache/` (365 MB) holds per-candidate, per-seed pickles (auc, audit, candidates, deep, diagnostic, drfe, fg, fs, ni, pu, reference, text, ctx.pkl). These allow blend and stacking experiments without retraining — directory listing; [experiments/common.py](experiments/common.py)
- Prices: `prcc_f` and `csho` are non-missing and positive for every row of the paper file (computed), which is why the market block (A1) could be built — computed (this review)

### Inferences
- Industry codes ARE obtainable: LM10X has SIC and FFInd per filing, joinable through farr gvkey→cik. Protocol 3b used FF48 (T2) only for linked rows and filled the others by random copying. About 20% of non-fraud firm-years do not link, which limits industry-based ideas.
- Because the XBRL extra tags hold SG&A, R&D, gross profit and operating income, a full Beneish SGAI or richer expense features is only possible for fiscal 2010+ (the fresh period), not for the 1991-2014 training data.
- Without CRSP there are no stock returns, volatility of daily returns, or trading volume beyond annual price x shares. Without Audit Analytics there are no auditor or restatement data. Without board or ownership data there is no governance. The report states that a WRDS subscription would be needed for audit and governance data.

### Gaps
- I did not verify LM10X coverage of the paper's firm-years beyond the protocol's numbers (linked share about 80% non-fraud / about 95% fraud in the validation years).
- No CIK column in the main file: linking depends entirely on farr. The completeness of farr for early-1990s firms was not checked here.
- The research_notes folder already contains `free_data_sources.md` from another researcher. It was not reviewed.

## Q4. Future work and open ideas already written down, and ideas explicitly untested

### Takeaway
The report's "Scope for future improvement" lists 8 items, none implemented: audit/governance data, a size-neutral model, a cost-based review budget, label-delay modelling, a fresh test period (since done, Protocol 7), stronger statistics, other markets, and LLM-proposed (CAAFE) features. The deck adds MD&A text with LLM embeddings, governance data, keeping labels current, and a size-neutral variant.

### Cited Findings
- Report section 12, ordered by "expected value per week of work", "None is implemented": 12.1 audit and governance data (audit fees, auditor changes, restatement history, board and ownership; needs WRDS; use the no-missingness and placebo design of Protocol 3b); 12.2 size-neutral model (drop log assets and log sales, or model within size buckets; judge mainly on Prec@1%); 12.3 cost-based review budget; 12.4 label-delay modelling (weight recent years by the probability their frauds are already discovered, a survival model of time to enforcement, instead of the hard 2010 cutoff); 12.5 fresh test period (later done in Protocol 7); 12.6 stronger statistics (DeLong per year, nested selection, more validation years); 12.7 other markets (India or EU); 12.8 CAAFE-style LLM-proposed features (not tried, needs an external API) — [report/Fraud_Detection_Project_Report.docx](report/Fraud_Detection_Project_Report.docx)
- The report says filing text and tone failed, so "the next data to add is structured, not textual" — [report/Fraud_Detection_Project_Report.docx](report/Fraud_Detection_Project_Report.docx)
- Deck slide 28 "Future work": add MD&A text (LLM embeddings) and board and auditor data; keep labels current; test a size-neutral variant because firm size leads SHAP. Speaker notes for slide 22 say the same — [presentation/Fraud_Detection_Midterm_IIITR.pptx](presentation/Fraud_Detection_Midterm_IIITR.pptx)
- Firm size dominates the tree SHAP (`log_sale`, `log_at` and their ranks), and a size-neutral model has not been tested — [README.md](README.md); [report/Fraud_Detection_Project_Report.docx](report/Fraud_Detection_Project_Report.docx)
- Explicitly listed as not tried: CAAFE / LLM features — [experiments/PROTOCOL_FG.md](experiments/PROTOCOL_FG.md). The FS protocol and README state the 10 families "are what was tried, not every method that exists" — [README.md](README.md)
- G1's top-of-list gains (NDCG@5%, Sens@5%) were "not tested further" — [README.md](README.md)
- Known limitations: Beneish approximations; 3 validation years; 2013-2014 too sparse under the paper's labels; labels are enforcement actions (undetected fraud = 0) — [README.md](README.md)

### Inferences
- Untested directions consistent with the data on disk: label-delay or survival weighting (12.4), which needs no new data; size-neutral or size-bucketed models (12.2); a top-of-list or cost objective (12.3); LLM-proposed features (12.8, needs an API); MD&A text embeddings, which would need new 10-K text downloads since only LM summary counts are on disk; using the completed labels (983 frauds) for training as well as evaluation.
- Ideas already tested that should not be re-proposed without a new angle: LM tone/word counts, FF48 industry-relative ratios, filing delay, amendments, PU loss, peer k-means deviations, modified-Jones accruals.

### Gaps
- The slides cite governance/text papers by reference number ([30], [31]). Their exact identities were not extracted here.

## Q5. Runtime, cost and hardware facts

### Takeaway
Everything runs CPU-only (16 logical cores on this machine). A full pipeline run with `--reuse` takes about 90-118 minutes, a full run from scratch about 2 hours, and `--fast` 30-45 minutes. Single tree members take about 2-3 minutes per test-year window, and deep feature-token models up to about 14 minutes per fold.

### Cited Findings
- README: `--fast` (2003-2008) about 30-45 min on a CPU; full run about 2 hours; `--reuse results` skips tuning steps 2-4, about 90 min — [README.md](README.md)
- Logged durations: `results_tt/log.txt` "Done in 89.5 min"; `results_labels/log.txt` "Done in 97.2 min" (reuse, skip_final); `results/log.txt` "Done in 118.2 min" (the SHAP rerun with reuse) — [results/experiments/log_pipeline_tt.txt](results/experiments/log_pipeline_tt.txt); [results_labels/log.txt](results_labels/log.txt); [results/log.txt](results/log.txt)
- The report says "about 2 hours on a laptop CPU" — [report/Fraud_Detection_Project_Report.docx](report/Fraud_Detection_Project_Report.docx)
- Deep models are "pure PyTorch and CPU-friendly". Mamba blocks are hand-written because mamba-ssm needs CUDA — [src/deep_models.py](src/deep_models.py)
- Per-task timings: an XGB member for one test year and seed 139-165 s, LR about 25 s (NI test); Temporal Mamba fold about 35 s; TabM fold 239 s; Mambular fold 858 s — [results/experiments/log_ni_test.txt](results/experiments/log_ni_test.txt); [results/experiments/log_deep_cv2.txt](results/experiments/log_deep_cv2.txt)
- TT SHAP audit took 18.5 min — [results/experiments/log_audit_temporal_shap.txt](results/experiments/log_audit_temporal_shap.txt)
- XBRL download: about 50 min, resumable, limited to 10 requests per second by the SEC — [experiments/fresh/xbrl_download.py](experiments/fresh/xbrl_download.py)
- Pinned environment: lightgbm 4.7.0, xgboost 3.4.1, torch 2.15.0, scikit-learn 1.9.1, shap 0.54.1, pandas 3.0.6, numpy 2.5.4. The saved model only loads with these versions — [requirements.txt](requirements.txt); [README.md](README.md)
- `nproc` reports 16 on this machine (this review). No GPU is mentioned anywhere in the project files.

### Inferences
- A full protocol, with a 3-seed screen plus test on 3 seeds over 10 windows, has historically meant several hours of CPU time per candidate family. The cache of stored scores makes blend-only ideas nearly free.

### Gaps
- No exact CPU model or RAM is recorded in the project files, and `wmic` returned nothing.

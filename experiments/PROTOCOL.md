# Experiment protocol (written before any candidate was run)

Reference model: the committed `PROPOSED: ensemble` = rank-averaged LGBM + XGB + LR on the 133
`enhanced_rank` features, with the hyper-parameters stored in `results/summary.json`.

Evaluation protocol is unchanged: expanding window, 2-year gap, serial-fraud masking, test years
2003-2014, years with < 15 frauds left out of means (so 2003-2012 is what counts).

## Candidates
1. **DRFE**: dynamic recursive feature elimination, re-run inside every training window.
2. **Sequence models**: FT-Transformer and Mambular (features as tokens), plus Temporal Transformer
   and Temporal Mamba (each firm's fiscal-year history as the sequence).
3. **From 2024-26 papers**: TabM (Gorishniy et al., ICLR 2025) and the focal PU risk
   iFPU (Zavitsanos & Paliouras, arXiv 2605.14467).

## Stage 1: validation (fiscal years up to 2001 only)
All choices (hyper-parameters, epochs, feature-selection settings, which members join an ensemble)
are made on the rolling-origin folds that validate on 1999, 2000 and 2001. A candidate advances to
an ensemble test only if its mean CV AUC beats the reference's by at least 0.002, the tolerance
`pick_combo` already uses. Sequence models are also scored on the test years as standalone models,
whatever their CV result, because those numbers were explicitly requested.

## Stage 2: test years, once per frozen candidate
Three seeds for the reference and for every candidate. Compared with the reference using the
project's paired bootstrap (1000 resamples, frauds clustered by AAER case, non-frauds by firm) on
AUC, NDCG@5% and Sens@5%, averaged over seeds inside each resample.

A candidate is **adopted** only if all of these hold on 2003-2012:
* **Significant**: at least one metric has P(gain <= 0) <= 0.05.
* **No harm**: no metric has P(gain <= 0) >= 0.90.
* **Persists over seeds**: the significant metric's gain is positive for all 3 seeds.
* **Persists over time**: the significant metric's gain is positive in both 2003-2008 and 2009-2012.

Holm-adjusted p-values across all candidates and metrics are reported as well. An adopted candidate
that does not survive Holm is labelled "promising, not confirmed" in the write-up.

No threshold, metric or candidate is changed after test results are seen. Anything that fails is
reported as a failure and left out of the committed pipeline.

## Amendment, 2026-11-03 (before any candidate was scored on test years)
A bug stopped the deep tuning runs at epoch 20, so the planned epoch 30 was never trained. The epoch
grid is now 5, 10, 15 and 20. FT-Transformer and Mambular had their best validation score at epoch
20 and were still rising, which points to under-training. New rule, decided on validation data only:
if an architecture's best score is at the last epoch of the grid, that trial is re-run on pre-2003
folds with epochs 30, 40, 50 and 60, and the best of all those scores is frozen.

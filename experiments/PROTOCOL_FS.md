# Protocol 2: dynamic feature selection (written 2026-11-04, before any of these candidates was run)

Goal: find a feature-selection method, re-run inside every training window, that raises test ROC AUC
over the current model. Evaluation protocol as in `PROTOCOL.md`: expanding window, 2-year gap,
serial-fraud masking, test years 2003-2012.

Reference: the current committed ensemble, LGBM + XGB + LR on 133 features + Temporal Transformer.
Its scores are already stored: CV AUC 0.7723 (seeds 43, 44), test AUC 0.7433 (3 seeds).

Status of the evidence: the 2003-2012 test years were already used for `PROTOCOL.md`, so these
results can be called **exploratory** at best, however strong they look. The selection rules below
limit the damage; they cannot make the test years fresh.

## Candidates
Every selector sees training-window rows only, with inner folds that copy the outer 2-year gap.

| Family | Method | Settings |
|---|---|---|
| F1 | Stability selection (Meinshausen & Buhlmann 2010): firm subsamples, out-of-time importance | keep at frequency 0.6, 0.8, 0.95 |
| F2 | Shadow-feature test (Boruta, Kursa & Rudnicki 2010) | beat shadow quantile 1.0, 0.95, 0.75 |
| F3 | Out-of-time permutation importance (after Altmann et al. 2010) | keep AUC drop > 0, or > 1 SE |
| F4 | Adversarial drift filter (Uber AVFS, arXiv 2004.03045) | drop top 10%, 20%, 30% drifting |
| F5 | mRMR, relevance / redundancy (Ding & Peng 2005) | keep 40, 70, 100 |
| F6 | DRFE for LR only; trees keep all features | tau 0.02, 0.05, 0.10 x best, one_se |
| F7 | Soft selection: each tree bag samples features by importance | alpha 0.5, 1.0 |
| F8 | DRFE-subset LightGBM added as an extra member | tau 0.02, 0.05, 0.10, one_se |
| F9 | Pool of 133 + 60 firm-history features, then DRFE | none, tau 0.02, 0.05 x best, one_se |
| F10 | Per-firm-year gates inside the Temporal Transformer (after TFT's variable selection network) | softmax, sigmoid x 68, 133 inputs |

F6 to F9 reuse the existing `DynamicRFE`. The Temporal Transformer stays in the ensemble for F1 to
F9, unchanged. F10 replaces it.

## Stage 1: validation only (fiscal years up to 2001)
1. **Screen** every setting on seed 42 against the reference on seed 42.
2. **Check** the best setting of each family on fresh seeds 43 and 44. A setting passes the gate if its
   mean CV AUC beats the reference's by at least 0.002 on those seeds.
3. **Advance** at most the 3 passing settings with the highest fresh-seed gain. Settings are then
   frozen. A combination of two passing families may take one of the 3 places only if it passes the
   same gate on seeds 43 and 44.

No test-year score of any candidate is computed before this step ends. If nothing passes, the test
years are not used at all.

## Stage 2: test years, once per frozen candidate
Same rule as `PROTOCOL.md`, against the reference above, on 2003-2012:
* **Significant AUC**: P(AUC gain <= 0) <= 0.05. AUC is the required metric here.
* **No harm**: no metric among AUC, NDCG@5%, Sens@5% has P(gain <= 0) >= 0.90.
* **Persists over seeds**: AUC gain positive for all 3 seeds.
* **Persists over time**: AUC gain positive in both 2003-2008 and 2009-2012.

Holm-adjusted p-values across the candidates and metrics tested here are reported as well. If more
than one passes, the one with the highest fresh-seed validation gain is adopted.

## Outcome
* A candidate passes: it goes into `run_pipeline.py` and the scorer, verified and committed, labelled
  exploratory, with the Holm result stated.
* Nothing passes: the pipeline stays as it is and the report says feature selection could not be made
  to raise ROC with these methods. The list above is what was tried. It is not every technique that
  exists, and the report says so.

No threshold, metric or candidate changes after test results are seen, and no further variant is
tried on these test years after a failure.

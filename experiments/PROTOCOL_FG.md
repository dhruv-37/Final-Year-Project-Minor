# Protocol 5: automatic feature generation (written 2026-11-10, before any candidate was run)

Goal: test whether features generated automatically inside every training window, rather than chosen
by hand, raise ROC AUC over the current model. Feature selection was tested in `PROTOCOL_FS.md`. This
protocol tests feature *construction*, which was never tried.

Reference: the committed model, LGBM + XGB + LR on 140 features (133 + 7 market ranks, recency weights,
retuned trees) + Temporal Transformer. CV AUC 0.7868 on seeds 43 and 44.

Evaluation as in `PROTOCOL.md`: expanding window, 2-year gap, serial-fraud masking, test years 2003-2012.
These test years have been used by four earlier protocols, so any result here is **exploratory**.

## Method: expand and reduce, after OpenFE (Zhang et al., ICML 2023)
Own implementation, because the published package does not respect time order. Everything below runs
again inside each training window and uses only that window's rows and labels, with the outer
serial-fraud mask applied.

**Expand.**
* Round 1, from the 28 raw items: every ratio a / b (756 ordered pairs) and every scaled difference
  (a - b) / total assets (378 pairs).
* Round 2, recursive, from the 25 best round-1 features: the change from the firm's previous consecutive
  year, the deviation from the firm's own mean over up to 3 earlier years, and every ratio of two of them.
  Round 2 only applies to G2.
* Every generated feature enters as a within-year percentile rank, like the existing rank features. That
  makes it scale-free and uses only the same fiscal year's cross-section and the firm's earlier years.
* There is no separate filter for duplicates of existing features. A candidate the base model already
  has gains almost nothing in step 2 of the reduce stage, so it does not survive.

**Reduce.** Two inner folds inside the window copy the outer gap: train on years up to L-3 and score L-1,
then train up to L-2 and score L, where L is the window's last year.
1. A base LightGBM on the 140 existing features gives each inner row a margin.
2. Each candidate gets a small LightGBM boosted from that margin on the candidate alone: 30 trees, 7
   leaves. Its score is the drop in inner-validation log loss, averaged over the two inner folds. This is
   OpenFE's FeatureBoost.
3. Successive halving: every candidate is scored with 25% of the negatives, then the top 20% is
   rescored with all rows. The best 100 go on.
4. A LightGBM on the 140 features plus those 100 ranks them by split gain. The top k are added, which
   is OpenFE's second stage. For G2, round 2's candidates go through steps 2 and 3 on their own, and
   this stage ranks round 1's 100 and round 2's 100 together.

Rows are sampled as the members do: all frauds and 20 non-frauds per fraud. Log loss is scored after
correcting the margin for that sampling.

The generated features go to LGBM, XGB and LR. The Temporal Transformer is unchanged.

## Candidates
| Family | Rounds | Settings |
|---|---|---|
| G1 | 1, raw-item ratios and differences | k = 10, 25, 50 |
| G2 | 2, round 1 plus the recursive temporal round | k = 10, 25, 50 |

## Stage 1: validation only (fiscal years up to 2001)
1. **Screen** every setting on seed 42 against the reference on seed 42.
2. **Check** the best setting of each family on fresh seeds 43 and 44. It passes the gate if its mean CV
   AUC beats the reference's 0.7868 by at least 0.002. The seed also changes the selection's subsampling.
3. **Advance** at most 2 passing settings, by fresh-seed gain. Settings are then frozen.

No test-year score of any candidate is computed before this step ends. If nothing passes, the test
years are not used.

## Stage 2: test years, once per frozen candidate
Against the reference on 2003-2012, adopt only if:
* P(AUC gain <= 0) <= 0.05;
* no metric among AUC, NDCG@5% and Sens@5% has P(gain <= 0) >= 0.90;
* the AUC gain is positive on all 3 seeds and in both 2003-2008 and 2009-2012.

Holm-adjusted p-values are reported. If both pass, the one with the higher fresh-seed gain is adopted.

## Outcome
* A candidate passes: it goes into `run_pipeline.py` and the scorer, labelled exploratory, with the Holm
  result stated.
* Nothing passes: the pipeline stays as it is, and the report says automatic feature generation was
  tried and did not raise ROC AUC with these methods. Two families of one method are not every technique
  that exists. LLM-proposed features (CAAFE, Hollmann et al. 2023) were not tried, because they need an
  external API.

No threshold, metric or candidate changes after results are seen. No further variant is tried on these
test years after a failure.

## Result, 2026-11-10: nothing passed, so the test years were not used
Screen, seed 42 (reference 0.7863):

| Setting | CV AUC | Gain |
|---|---|---|
| G1, k = 10 | 0.7863 | −0.0000 |
| G1, k = 25 | 0.7865 | +0.0002 |
| G1, k = 50 | 0.7836 | −0.0027 |
| G2, k = 10 | 0.7842 | −0.0021 |
| G2, k = 25 | 0.7868 | +0.0005 |
| G2, k = 50 | 0.7837 | −0.0026 |

Check, fresh seeds 43 and 44 (reference 0.7868, gate +0.002):

| Setting | CV AUC | Gain | NDCG@5% | Sens@5% | Gate |
|---|---|---|---|---|---|
| G1, k = 25 | 0.7859 | −0.0009 | 0.215 (ref 0.207) | 0.299 (ref 0.284) | Fail |
| G2, k = 25 | 0.7850 | −0.0018 | 0.210 | 0.277 | Fail |

G1 raised NDCG@5% and Sens@5% on the fresh seeds. The protocol decides on AUC, so this does not change
the verdict. It was not tested on the test years and is not claimed.

**Descriptive, after the result, so not a check:** the selected features are unstable
(`results/experiments/fg_selection_stability.json`). Comparing top-25 sets:
* G1, same window with another seed: Jaccard overlap 0.17. Same seed, another window: 0.06.
* G2: 0.08 and 0.02.
* No generated feature was chosen in all 9 runs (3 windows x 3 seeds) of either family.

This fits the gains being at noise level: about 900 frauds leave too little signal to pick ratios
reliably out of more than a thousand candidates.

# Protocol 6: is dynamic feature selection no worse? (written 2026-11-10, before this candidate ran)

Question: can features be selected inside every training window at no measurable cost in ROC AUC?
`PROTOCOL_FS.md` asked whether selection *improves* AUC, and no method passed. This protocol asks a
different question, non-inferiority, and was written after seeing those validation results, which is why
it is reported as a separate, exploratory test.

Reference: the committed model, LGBM + XGB + LR on 140 fixed features + Temporal Transformer. Its CV AUC
is 0.7868 on seeds 43 and 44. Its scores are already stored.

Evaluation as in `PROTOCOL.md`: expanding window, 2-year gap, serial-fraud masking, test years 2003-2012,
3 seeds, paired bootstrap clustered by AAER case and firm, 1,000 resamples. The test years have been used
before, so the result is **exploratory**.

## Candidate, frozen
The best setting of `PROTOCOL_FS.md` (F9_0.02_best), unchanged except that the pool now holds the
features the current model uses:
* **Pool:** the 140 features plus 60 firm-history features. For 20 rank features, each gets the deviation
  from the firm's own mean over its previous 4 years, that deviation over the firm's own volatility, and
  the volatility. That makes 200 candidates.
* **Selection, inside every training window:** dynamic recursive feature elimination (`DynamicRFE`, tau
  0.02, rule "best") on three inner folds that keep the 2-year gap. Importance is a consensus of LightGBM
  SHAP and logistic regression coefficients, recomputed each round. Its internal LightGBM settings are the
  ones frozen in `PROTOCOL_FS.md`.
* **Members:** LGBM, XGB and LR are fitted as in the reference (same settings, recency weights and seeds)
  on the selected features only. The Temporal Transformer is unchanged.

Selection is seeded with the run's seed, so each seed selects on its own.

## Margin
**delta = 0.003 AUC**, below 7% of the model's gain over the paper's baseline (+0.0435). Losing less
than this keeps more than 93% of that gain.

## Stage 1: validation, seeds 43 and 44
Continue to the test years only if the mean CV AUC change is above -0.003. This is a safety check, not
evidence.

## Stage 2: test years 2003-2012, seeds 42, 43 and 44
The AUC change is averaged over the 3 seeds, with a 95% CI from the paired bootstrap. The verdict follows
fixed rules:
| Result | Verdict |
|---|---|
| Lower CI bound above -0.003 | **Non-inferior**: dynamic selection costs less than 0.003 AUC |
| Lower CI bound above 0 | Non-inferior and superior |
| Upper CI bound below 0 | **Measurable cost**: report it with its CI |
| Otherwise | **Inconclusive**: the CI is too wide to tell |

NDCG@5% and Sens@5% are reported with their CIs, Prec@1% without one, and the change in each
sub-period (2003-2008 and 2009-2012) as well. They do not change the verdict.

## Outcome
* **Non-inferior:** dynamic selection goes into `run_pipeline.py` and the scorer, because the project
  owner asked for it. It is described as "no measurable loss", never as an improvement unless the
  superiority row applies.
* **Otherwise:** the pipeline keeps the fixed pool, and the measured cost or the inconclusive interval
  is reported.

No threshold, margin or candidate changes after results are seen.

## Amendment, 2026-11-10, before any result
The first run crashed. In the earliest inner folds, the 3-year market-growth rank has no values, and
logistic regression's imputer drops such a column, so the importance vectors had 199 and 200 entries.
`_lr_importance` now gives such a column 0 importance. Where no column is empty, as in every run of
`PROTOCOL_FS.md`, the result is unchanged. This is a code fix, not a change to the method.

## Result, 2026-11-10
**Stage 1** (`results/experiments/cv_ni.json`): CV AUC 0.7843 against 0.7868, a change of −0.0025,
above −0.003, so the test years were run.

**Stage 2** (`results/experiments/ni_test.json`), 2003-2012, mean of 3 seeds:
| Metric | Fixed 140 | Dynamic | Change | 95% CI |
|---|---|---|---|---|
| ROC AUC | 0.7484 | 0.7493 | +0.0008 | −0.0043 to +0.0062 |
| NDCG@5% | 0.0836 | 0.0817 | −0.0020 | −0.0146 to +0.0080 |
| Sens@5% | 0.1571 | 0.1574 | +0.0003 | −0.0236 to +0.0192 |
| Prec@1% | 0.0188 | 0.0197 | +0.0009 | not computed |

AUC change by seed: +0.0022, +0.0003, +0.0001. By period: 2003-2008 −0.0006, 2009-2012 +0.0031.

**Verdict: inconclusive.** The lower bound, −0.0043, is below −0.003 and the upper bound is above 0. The
point estimate is close to zero, but the test cannot rule out a loss larger than the margin. As fixed
above, the pipeline keeps the fixed pool.

Described after the result, not part of the verdict: the number of chosen features ranges from 59 to 200
across windows and seeds (mean 134, of which 33 are history features), and in 2005 the three seeds chose
59, 200 and 111. The validation AUC along the elimination path is nearly flat, so the "best" point
moves with the seed. The subsets are nested, so seeds overlap on the smaller set (mean Jaccard 0.70),
and 45 features were kept in all 30 runs.

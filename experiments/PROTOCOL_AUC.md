# Protocol 3: raising ROC AUC with the existing data (written 2026-11-04, before these candidates ran)

Target stated by the project owner: test AUC above 0.8. Reference: the committed ensemble, LGBM + XGB +
LR on 133 features + Temporal Transformer, test AUC 0.7433, CV AUC 0.7723 (seeds 43, 44).

The target is not an acceptance rule. Reaching 0.8 by leakage would be worthless, so the evaluation is
fixed: expanding window, 2-year gap, serial-fraud masking, test years 2003-2012. The 2003-2012 test
years have been used before, so any result here is **exploratory**.

## Candidates
| Family | Idea | Settings |
|---|---|---|
| A1 | Market features from price and shares, within-year ranks (Bertomeu et al. 2021 find market variables matter) | 7 extra features |
| A2 | Peer deviations: per-year k-means peers stand in for the missing industry codes | 20 extra features |
| A3 | Abnormal accruals: modified-Jones style residuals within peer groups | 4 extra features |
| A4 | Recency weighting of tree training rows | half-life 3, 6 years |
| A5 | Different learners added to the ensemble | random forest, extra trees, both |
| A6 | Larger Temporal Transformer ensemble | 6 networks |
| A7 | Unequal blend weights, using stored scores | TT x 2, TT x 3, no LR, TT x 2 without LR, + Temporal Mamba |
| A8 | Longer firm history in the Temporal Transformer | 8 years |
| A9 | Year-aware feature modulation in the Temporal Transformer (after arXiv 2512.03678) | input level |
| A10 | Surprise features: actual minus predicted from the firm's prior year, fit on non-fraud training rows | 20 extra features |
| A11 | Re-tuning trees for the full ensemble rather than alone | 40 trials each: LGBM, XGB, both |
| A12 | Direct AUC optimisation: pairwise loss in the Temporal Transformer | pairwise logistic |
| A13 | Covariate-shift weights from a train-versus-evaluation-year classifier, no labels used | clip 5, 10 |

Everything uses only rows and fiscal years available at that window. A13 reads the evaluation year's
features but never its labels, which matches deployment, where the market cross-section is known.

## Stage 1: validation only (fiscal years up to 2001)
Screen every setting on seed 42. Check each family's best setting on fresh seeds 43 and 44. A setting
passes if its mean CV AUC beats the reference by at least 0.002 on those seeds. Passing settings that
do not conflict may be combined; a combination passes the same check. At most 3 settings advance, by
fresh-seed gain. If none passes, the test years are not used.

## Stage 2: test years, once per frozen candidate
Adopt only if, on 2003-2012 against the reference: P(AUC gain <= 0) <= 0.05; no metric among AUC,
NDCG@5%, Sens@5% has P(gain <= 0) >= 0.90; AUC gain is positive on all 3 seeds and in both 2003-2008
and 2009-2012. Holm-adjusted p-values are reported. If several pass, the one with the highest
fresh-seed gain is adopted.

## Reporting
The final AUC is reported whether or not it reaches 0.8, with its label. A separate diagnostic may show
what the score becomes without serial-fraud masking and with a random split, to explain the higher
figures seen in some papers. Those numbers are never candidates.

No rule changes after test results are seen.

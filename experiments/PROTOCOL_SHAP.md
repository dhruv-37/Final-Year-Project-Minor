# Protocol 4: explaining the Temporal Transformer (written 2026-11-04, before any attribution was computed)

Goal: extend SHAP to the Temporal Transformer member, the one part of the deployed ensemble it did not
cover, and show that the explanations can be trusted. No model is chosen or changed and no test metric is
used to select anything, so scores must stay exactly as they are.

## Method
Expected gradients (Erion et al. 2021), the method behind `shap.GradientExplainer`: integrated gradients
(Sundararajan et al. 2017) averaged over a reference sample, which approximates SHAP values.

* **Explained output:** each network's logit, averaged over the 3 networks, in the same log-odds units as
  the tree SHAP values. The deployed member averages probabilities, so check B compares the two.
* **Explained rows:** the same 3,000 firm-years of 2012-2014 as the tree members. These years are out of
  sample for the deployed model, which trains on 1991-2010.
* **Reference:** 50 firm-years with a complete 5-year history, drawn at random from the training rows.
* **Inputs:** 5 years x 68 features. Only observed values are attributed. A row's missing-value flags and
  absent years stay as they are, so they get no attribution. The method explains the values a firm
  reported, given the history it has.
* **Grouping:** cells add up, so a feature's attribution is its sum over years and a year's is its sum over
  features.

## Checks, with thresholds fixed in advance
| | Check | Pass if |
|---|---|---|
| A | Known answer: on a linear network, attributions equal w(x - mean reference) | max error below 1e-4 |
| A | Missing values and absent years get no attribution | exactly 0 |
| B | Output: the network with gradients on matches the deployed network | max difference below 1e-4 |
| B | Mean logit ranks firms like the deployed mean probability | Spearman at least 0.99 |
| C | Completeness: attributions sum to output minus reference output | median error at most 1% of the output's SD, 99th percentile at most 5% |
| D | Second method, permutation Shapley on 300 rows: global feature importance | Spearman at least 0.80 |
| D | The same, feature attributions within a firm-year | median Pearson at least 0.80 |
| D | The same, each year's share of attribution (exact Shapley over 5 years) | largest gap at most 0.05 |
| E | Stability across reference samples: global feature importance | Spearman at least 0.95 |
| E | Stability across the 3 networks: global feature importance | mean pairwise Spearman at least 0.70 |
| F | Faithfulness: resetting a row's top 20 positive cells to the reference lowers the logit more than resetting 20 random observed cells | at least 2 times the drop, and a row-bootstrap 95% CI of the difference above 0 |
| G | Sanity, re-initialised weights (Adebayo et al. 2018): global importance against the real model | absolute Spearman at most 0.30 |
| G | Sanity, a network trained on labels shuffled within each year | absolute Spearman at most 0.50 |
| H | Repeat run with the same seed | max difference below 1e-3 |

Failing a check is reported, with what it means for the claim. Thresholds do not change after results are
seen. If D or E fails, the explanations are described only as far as the checks that passed allow.

## No regressions
A full pipeline rerun must leave every model score in `test_predictions.csv` unchanged, apart from the small
Temporal Transformer differences already seen between runs, which are reported if they occur. The tree and
logistic regression SHAP values must also stay the same.

## Amendment, 2026-11-04, after the audit ran
Check G failed: the re-initialised network's global importance has Spearman 0.425 with the real model's,
above the limit of 0.30. The verdict and the threshold stand. A follow-up diagnostic, chosen after seeing this
result and so not a pass/fail check, looks for the reason: `experiments/audit_temporal_shap_g.py` uses 10
random networks, tests whether importance simply follows how much each input varies, and tests whether
cells picked by a random network's attributions change the real model's output.

**Result of the follow-up, 10 random networks:**
* Global importance against the real model: mean Spearman 0.348 (range 0.223 to 0.511), top-15 overlap 6.6 on
  average. A network with no training shares part of the real model's global ordering.
* My hypothesis was wrong. Removing how much each input varies barely changes this (0.322), and importance
  does not follow that spread (-0.325 for the real model).
* Importance does follow how often a feature is observed (0.643 for the real model). The diagnostic did not
  test whether this explains the overlap, so the cause of the failure remains unproven.
* Within a firm-year, a random network's attributions are unrelated to the real model's: median Pearson -0.031
  (range -0.185 to 0.066).
* Resetting the 20 cells a random network ranks highest changes the real model's logit by +0.087 on average,
  so it does not lower it. Resetting the 20 the real attributions rank highest lowers it by 0.962.

**What may be claimed:** explanations of a firm-year depend on the trained model and change its output. The
global feature ranking is stable and agrees with a second method, but it failed the randomisation check, so it
is described as indicative and not as what the model learned. The follow-up does not reverse the failure, and
the audit as a whole did not pass.

## Amendment, 2026-11-04, after the pipeline rerun
Tested against `No regressions`:
* All model scores in `test_predictions.csv` are identical, including the Temporal Transformer's, with weights
  identical as well. Every metric file is identical. The tree SHAP columns are identical.
* **The logistic regression SHAP column changed, by up to 0.005 (Spearman 0.999, same top 10).** So this one
  criterion failed. Cause: `shap.PermutationExplainer` calls `np.random.seed(None)` when it is given no seed,
  which reseeds numpy from the operating system, so this column already differed on every run. It is only an
  importance column and is not used by any plot or score. `member_shap` now passes a fixed seed, and two runs
  now give the same values. This column's values changed once more as a result.
* The explanations were regenerated from the saved model, not by retraining: `results/final_models.joblib` is
  the audited model, and the attributions the pipeline writes match those the audit tested to 5e-9.
* A labelling error found on inspection and fixed: every Temporal Transformer input was labelled as a rank, but
  `loss` and `log_at` are not. Values were not affected.

## What this does not claim
SHAP shows what the model uses, not why a firm committed fraud. Attributions from correlated years can be
split between those years somewhat arbitrarily, which is why check E is reported. Members are rank-averaged,
so SHAP values from different members cannot be added together. What can be stated exactly is each member's
share of the final score: the score is the mean of the four within-year ranks.

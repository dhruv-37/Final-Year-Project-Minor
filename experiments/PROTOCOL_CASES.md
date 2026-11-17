# Protocol 9: new versus continuing frauds in the test years (written 2026-11-17, before any score is split)

Question: the test years contain frauds whose AAER case began years earlier. Their earlier years are in
the training window, masked to "clean" by serial-fraud masking, but the firm's features (and, for the
Temporal Transformer, its history) are still seen. Does the model detect frauds that start in the test
year, or mainly frauds already under way? Where do the two adopted gains (firm history; market features
with recency weights) come from?

Only counts were looked at before this was written: of the 358 test frauds in 2003-2012, the first
labelled year of the case in the data is 2 or more years before the test year for 181, one year before
for 71, and the test year itself for 106.

## Groups
For a fraud firm-year in test year T, the case start s is the earliest fiscal year of its AAER case
(`p_aaer`), taken as the minimum over the paper's AAER file (`AAER_firm_year.csv`, which also lists
firm-years missing from the data) and the labelled rows of the data. 13% of test frauds have a case id
that is not in the AAER file; for them the data alone gives s.
* **Continuing:** s <= T-2 (the case has years inside the training window).
* **Gap-start:** s = T-1.
* **New:** s = T.

## Scores and metrics
The saved test predictions of the final run (`results/test_predictions.csv`, seed 42), not retrained.
For each group, every fraud of the group is compared with all non-frauds of its own test year:
* **AUC:** the share of (group fraud, same-year non-fraud) pairs ranked correctly, pooled over 2003-2012
  (the yearly AUCs weighted by the group's frauds that year). No minimum-fraud rule, since group-years are
  small.
* **Sens@5%, Sens@1%:** the share of the group's frauds inside their year's top 5% / 1% (the top is set
  by all rows of the year, as in the main results).
* Models: the baseline RUSBoost, the proposed ensemble, the ensemble without firm history, the previous
  ensemble (no market features or recency weights), and the Temporal Transformer alone.
* Gains with 95% intervals from a paired bootstrap (1,000 resamples; inside each year, frauds resampled by
  AAER case and non-frauds by firm, as in the project): proposed vs baseline, proposed vs without firm
  history (the firm-history gain), proposed vs previous ensemble (the market/recency gain).

## Readings fixed in advance
* The gain over the baseline **holds on new cases** if its AUC interval on the new group is above zero.
* A component's gain **comes mainly from continuing cases** if its AUC gain on the continuing group is
  positive with an interval above zero while the new-group interval includes zero or is negative.
* Sensitivity: the same table with s taken from the data alone.

Nothing is retrained and no label is changed. The test years have been used many times, so all results are
descriptive.

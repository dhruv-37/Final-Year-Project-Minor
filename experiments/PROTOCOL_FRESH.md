# Protocol 7: does the saved model hold on a fresh period and a fresh data source? (written 2026-11-13, before any fresh-year score was computed)

Question: the paper's dataset ends at fiscal 2014 and every result so far comes from it. Does the saved
model still rank frauds above non-frauds on fiscal years it has never touched, built from a different
source?

## Model, frozen
`results/final_models.joblib` as committed: LGBM + XGB + LR on 140 features + Temporal Transformer,
trained on fiscal 1991-2010. Nothing is refit, re-tuned or re-weighted. The first fresh test year is
2015, so the 2-year gap of `PROTOCOL.md` holds with room to spare. No training label is later than 2010,
so serial-fraud masking has nothing to mask.

## Fresh data (none of it is in the paper's file)
* **Financials:** every 10-K filer in the SEC's XBRL API (`data.sec.gov`), fiscal 2010-2020. Each number
  is the one in the earliest 10-K that reports it, never a later restated comparative
  (`fresh/xbrl_download.py`). Tags are mapped to the 28 Compustat items and the 14 paper ratios are
  rebuilt from them (`fresh/xbrl_build.py`, `fresh/ratios.py`).
* **Universe:** filer-years with a 10-K, a classified balance sheet and the core items (total assets,
  current assets and liabilities, total liabilities, common equity, cash, retained earnings, net income).
  Within-year ranks and the final rank average use this population.
* **Labels:** `fresh/aaer_labels_fresh.csv`, written by hand from the text of the AAERs released
  2017-2024. A firm-year is a fraud if a release says the firm's ANNUAL financial statements for that
  year were materially misstated. Tier A: the SEC describes fraud. Tier B: a material misstatement without
  a fraud finding. Primary label = A or B, the definition of Dechow et al. (2011) that the paper's labels
  follow. Quarter-only, disclosure-only, internal-control-only and bribery cases are not labelled.
  The labels were finished before any model score existed.

## Known weaknesses, stated in advance
* XBRL is not Compustat. Filers pick their own tags, cost of goods sold and depreciation are missing for
  some filers, and there is no share price: the price is public float / shares from the 10-K cover.
* The population is wider than Compustat's (more shells and unlisted registrants).
* Labels are incomplete for recent years. Releases from 2025-2026 are not covered and enforcement takes
  years, so unlabelled frauds sit among the non-frauds. This lowers measured AUC.
* The labels were written by the same assistant that built the test, from one reading, with no second
  coder.

## Step 1: bridge check (old years, both sources)
Fiscal 2012-2014 exist in both sources and are out of sample for the saved model. On the firm-years
present in both, the saved model is scored twice, once on Compustat inputs and once on XBRL inputs, with
the paper's labels. The difference is the cost of the data source alone. It is reported whatever it is,
and it does not change Step 2. These years were test years before, so this step is descriptive.

## Step 2: fresh test, run once
* **Primary years:** fiscal 2015-2018. Enforcement lag is at least six years there, as it was for the
  paper's 2003-2008 window. Fiscal 2019-2020 are reported separately and marked "labels incomplete".
* **Primary metric:** ROC AUC, the mean of the yearly AUCs over the primary years with at least 15 frauds
  (the project's rule). The pooled AUC over 2015-2018, computed on each year's risk percentile, is
  reported next to it, with NDCG@5% and Sens@5%.
* **Uncertainty:** the project's bootstrap, 1,000 resamples, frauds clustered by company, non-frauds by
  firm, seed 0.

## Verdict, fixed now
The reference is the saved model's family on 2003-2012: AUC 0.748, 95% CI [0.725, 0.772].
* **Holds:** fresh AUC is at least 0.725.
* **Weaker but real:** fresh AUC is below 0.725 and the lower end of its 95% CI is above 0.5.
* **Fails:** the 95% CI includes 0.5.

## Sensitivity checks, fixed now, reported whatever they show
1. Tier A only (tier B firm-years removed from the evaluation).
2. Only filer-years with a price (public float and shares on the cover), the closest match to the
   paper's sample, which needs every raw item.
3. Only filer-years with total assets of at least $10 million.
4. Each member alone (LGBM, XGB, LR, Temporal Transformer).
5. Quarter-only cases counted as frauds too.

Nothing in the build, the labels, the universe or the metric is changed after the fresh scores are seen.
If something has to change (a bug), the change and the scores before and after are both reported.

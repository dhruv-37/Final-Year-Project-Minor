# Protocol 8: completing the fraud labels for fiscal years up to 2014 (written 2026-11-17, before any re-scoring)

Question: the paper's labels come from AAERs released up to 2018 (the newest case in the data is AAER 3996).
Fraud keeps being discovered for years, so some firm-years in fiscal 2011-2014 that are labelled clean were
in fact misstated. How do the test results change when those labels are completed?

## New labels
* **Source 1:** the paper's own hand-collected file (`AAER_firm_year.csv`, AAERs up to 4012, Dec 2018).
  Cases newer than 3996 that are missing from the data are added.
* **Source 2:** AAERs released 2019-2024 (texts mirrored on Hugging Face, as in `PROTOCOL_FRESH.md`), read
  by hand. Same rule as the fresh labels: a firm-year is fraud if a release says the firm's annual
  financial statements for that year were materially misstated (tier A: fraud found; tier B: material
  misstatement without a fraud finding). Quarter-only, disclosure-only and bribery cases are not labelled.
* Fiscal years 1991-2014 only, the years in the paper's file. A firm-year is labelled only if it is
  in the data; firms are linked by CIK through the `farr` GVKEY-CIK table, within the link's dates.
* The case id for serial-fraud masking is the AAER number of the first release about the case.
* The labels are written before any score is computed on them. The original file is not changed:
  the completed labels are a separate file, `experiments/labels/labels_completed.csv`.

## Step 1: re-score, no retraining
The saved test-year predictions (`results/test_predictions.csv`, final run, seed 42) are scored against
the completed labels. Training is unchanged, so this measures only how the evaluation changes.
Reported for the proposed ensemble and the baseline:
* yearly AUC for 2003-2014 on both label sets;
* the mean over 2003-2012 (the project's period) on both label sets;
* the mean over every year that reaches at least 15 frauds after completion, if 2013 or 2014 qualifies;
* the gain of the proposed model over the baseline with the project's paired bootstrap (1,000 resamples,
  clustered by case and firm), on the completed labels.

Nothing about the labels is changed after the re-scored results are seen. These test years have been used
many times, so the results are descriptive.

## Step 2 (only if step 1 is done and the user asks): retraining
Completed labels also change training. For test years up to 2012 the model trains on years up to 2010,
where few labels change, so the effect is expected to be small; this is stated in advance.

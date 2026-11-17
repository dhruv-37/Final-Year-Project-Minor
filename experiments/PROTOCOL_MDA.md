# Protocol 10: MD&A text content (written 2026-11-18, before any MD&A feature is computed)

Question: does what management writes in the MD&A (Item 7 of the 10-K) add to the final model? Protocol 3b
tested Loughran-McDonald word counts and tone (fresh-seed gain −0.0002). This tests content: what the
MD&A talks about, how far it departs from peers, and how much it changed from the prior year (after
Brown, Crowley & Elliott, JAR 2020; Hoberg & Lewis, JCF 2017; Brown & Tucker, JAR 2011).

Reference: the committed final ensemble (`A1_mkt+A4_3+A11_both` members + Temporal Transformer, scores
cached by `experiments/run_auc.py`). Evaluation unchanged: expanding window, 2-year gap, serial-fraud
masking. Test results are exploratory, as in every protocol since Protocol 3.

## Text
* Source: EDGAR-CORPUS (huggingface.co/datasets/eloukas/edgar-corpus, Loukas et al. 2021, Apache-2.0),
  Item 7 of every 10-K filed 1993-2020, one per CIK and filing year (`experiments/mda/download_mda.py`).
* Download rules (fixed 2026-11-18, before any MD&A was linked): a stalled transfer is aborted and resumed,
  each attempt has a hard timer, and a source file not complete within 60 minutes is skipped with its
  filing year; skipped years are retried twice. A filing year still missing after that is left out, its
  firm-years count as having no MD&A (so they get donor values), and the report names the year.
* Fiscal year: the corpus gives only the filing year. Each MD&A is matched to the original 10-K of the
  same CIK and filing year in the Loughran-McDonald 10-X file (forms 10-K, 10-K405, 10-KSB, 10-KSB40,
  10-KT), whose period of report gives the fiscal year by the Compustat rule (a period ending January to
  May belongs to the previous year). The filing must be 0 to 540 days after the period end. If a CIK has
  more than one original 10-K in that filing year, the MD&A is dropped.
* Firm link: CIK to GVKEY through the `farr` table, the link valid on the filing date with 90 days of
  tolerance (as Protocol 3b).
* An MD&A with fewer than 1,000 words is treated as missing (it is usually "incorporated by reference" to
  an exhibit the corpus does not hold). Only the first 10,000 words are used.

## Features (added to LightGBM, XGBoost and logistic regression; the Temporal Transformer is unchanged)
* Vocabulary: unigrams and bigrams, the 20,000 most frequent with document frequency of at least 20,
  chosen once from a random sample of 8,000 MD&As of fiscal years up to 1997 (the training years of the
  first validation fold; a sample keeps memory in bounds), so no window sees later vocabulary. Tokens are
  lower-case words of at least 3 letters, English stop words removed.
* **M1, content (50):** TF-IDF, with the IDF and a 50-component truncated SVD fitted on the training
  window's MD&As only, then applied to the evaluation year.
* **M2, change and peers (3):** log word count; cosine similarity to the same firm's MD&A of the previous
  fiscal year (TF-IDF with IDF from fiscal years up to 1997); cosine distance to the centroid of the same
  fiscal year and 2-digit SIC (SIC from the Loughran-McDonald file; groups under 5 MD&As use the year's
  centroid). These use only the same or earlier fiscal years. An MD&A with no previous-year MD&A of the
  same firm takes the fiscal year's median similarity (no flag, no NaN).
* Settings: M1, M2, M1+M2.

## Missing text
About 13% of firm-years do not link to a CIK, more of them non-fraud, and a share of MD&As are short, so
"has text" may itself predict fraud. As in Protocol 3b: a firm-year without a usable MD&A gets the whole
block copied from a random firm-year of the same fiscal year that has one (fixed seed); fiscal years with
fewer than 500 usable MD&As (before EDGAR) draw from fiscal 1995-1997, as in Protocol 3b. There is no flag
and no NaN. Before modelling, the flag-only AUC of "has a usable MD&A" is reported.

## Stage 1: validation only (fiscal years up to 2001)
Screen M1, M2, M1+M2 on seed 42; check the best on seeds 43 and 44. It passes only if, on the mean of
seeds 43-44:
* CV AUC gain over the reference is at least +0.002 on all rows;
* the gain is at least +0.002 when the validation years are scored on firm-years with a usable MD&A only;
* a placebo (the block shuffled among firm-years with a usable MD&A within each fiscal year, then refit)
  gains at most +0.001.
The CV gain on new-case frauds (case starting in the validation year, as in Protocol 9) is reported but is
not a gate. If nothing passes, the test years are not used.

## Stage 2: test years, once
The adoption rule of Protocol 3 on 2003-2012 (3 seeds, paired clustered bootstrap, both sub-periods, no
harm), plus a positive AUC gain on firm-years with a usable MD&A, plus the Protocol 9 split of the gain by
new and continuing cases. Holm-adjusted p-values over all candidates tested on the test years are reported.

Caveat kept in the report: an MD&A can mention an investigation already under way. A user would see the
same filing, so this is not leakage, but part of any gain may be early detection rather than prediction.
No rule changes after test results are seen.

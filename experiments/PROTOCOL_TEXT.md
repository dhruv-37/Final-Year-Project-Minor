# Protocol 3b: filing data (written 2026-11-04, before any of these features was computed)

Adds external data to Protocol 3: the Loughran-McDonald 10-X summaries (Notre Dame) linked through the
GVKEY-CIK table of the R package `farr`. Same reference (test AUC 0.7433, CV 0.7723), same evaluation.
Test results are exploratory, as in Protocol 3.

## Known trap: linkage is itself a fraud signal
Before any modelling, about 98% of fraud firm-years link to a filer but only about 80% of the others.
"Has a filing" therefore predicts fraud with no filing content, probably through survivorship of the
link table. A gain from that would be fake. So:
* **No missingness signal.** Unlinked firm-years get values copied from a random linked firm-year of the
  same fiscal year. There is no flag and no NaN, so nothing marks a row as unlinked.
* **Linked-only gate.** The gain must also hold on linked firm-years alone.
* **Placebo gate.** Filing features are shuffled among linked firm-years within each fiscal year, then the
  candidate is refit. Its gain must be at most +0.001. A larger gain would mean something other than the
  filing content is helping.

## Timing rules
* Original filings only: 10-K, 10-K405, 10-KSB, 10-KSB40, 10-KT. Amendments carry restatements, so they
  never supply a firm-year's features.
* Compustat convention: a period ending in January to May belongs to the previous fiscal year.
* A filing counts only if it was filed 0 to 540 days after the period end. The earliest filing is used.
* T4 counts amendments filed before this firm-year's own filing date only.
* Link dates must cover the filing date, with 90 days of tolerance.

## Candidates (added to the trees and logistic regression; the Temporal Transformer stays unchanged)
| Family | Features |
|---|---|
| T1 | Text: size, vocabulary and tone shares, within-year ranks and change from the firm's prior filing |
| T2 | Industry: key ratios minus their Fama-French 48 industry-year median, and the industry's training fraud rate |
| T3 | Timing: filing delay rank, late-filing flag, small-filer form flag |
| T4 | Amendments filed in the previous 3 years |

Settings: T1, T2, T3, T4, T1+T2, T1+T2+T3, T1+T2+T3+T4.

## Stages
Stage 1 is validation only: screen on seed 42, then check the best setting of T1, T2, T3, T4 and the best
combination on seeds 43 and 44. A setting passes if it gains at least +0.002 CV AUC on all rows, gains at
least +0.002 on linked rows, and its placebo gain is at most +0.001. If none passes, the test years are
not used.

Stage 2 uses the adoption rule of Protocol 3, on all rows of 2003-2012, plus a positive AUC gain on linked
rows. Holm-adjusted p-values are reported across the candidates of Protocols 3 and 3b tested on the test
years.

## Amendment, 2026-11-04 (validation results seen, no test-year score computed)
T2 lost 0.0115 CV AUC on the screen, far more than an added feature block should. Its industry fraud rate
is computed from the same training rows it is then fed back to, which can overfit. Two variants join the
screen: T2n (no rate) and T2p (each row sees only labels at least 2 fiscal years older). The check step
takes the best of T2, T2n and T2p. New combinations using them also join the screen.

## Caveat kept in the report
A filing can mention an investigation already under way, so some text signal may reflect fraud being
discovered, not fraud being predicted. A real user sees the same filing, so this is not leakage, but the
two should not be confused.

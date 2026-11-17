# Free data sources and feature families for AAER fraud prediction (extending Bao et al. 2020)

Scope note: research budget was about 15 tool calls. The findings below come from search-result summaries plus one direct fetch of an SEC page. Several papers were seen only through abstracts or secondary summaries, and this is flagged where it applies. Coverage years for some SEC products are given from background knowledge; they are marked "unverified" and listed under Gaps.

## Q1. Which free sources exist, what years do they cover, how do you get them, and can they be linked to CIK?

### Takeaway
Almost every free non-financial source starts between 2003 and 2009 (insider XML, comment letters, EDGAR logs, 8-K item codes, XBRL). Only 10-K text on EDGAR (from the mid-1990s) covers most of the 1991-2014 window. Every EDGAR-derived source is keyed by CIK, so it links through the farr GVKEY-CIK table the project already uses. The problem is time coverage, not linkage.

### Cited Findings
- **SEC Insider Transactions Data Sets (Forms 3/4/5):** quarterly ZIP files flattened from the XML part of Forms 3, 4 and 5, "as filed". The download page is titled "January 2006 - September 2026", the earliest file is 2006 Q1 and the latest is 2026 Q3. Files are about 7.6-16.5 MB each, and a separate PDF documents the table definitions — [SEC Insider Transactions Data Sets](https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets)
  - An older data.gov catalog entry lists distributions from 2006 Q3 — [data.gov catalog](https://catalog.data.gov/dataset/insider-transactions-data-sets)
  - Commercial vendors claim coverage of all Form 3/4/5 filings since 2003, which is not free. This suggests 2003-2005 XML filings exist on EDGAR but are not in the SEC flat files — [sec-api.io insider API](https://sec-api.io/docs/insider-ownership-trading-api)
  - The SEC does not guarantee accuracy; the data come from filers and may contain extraction errors — [SEC page](https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets)
- **EDGAR Log File Data Sets (investor attention / downloads):** the first set covers 1 Jan 2003 to 30 Jun 2017. A second set covers 19 May 2020 to 30 Jun 2025. Data for July 2017 to May 2020 is no longer available. Files are CSV, one zip per day, often more than a million rows each — [SEC EDGAR log file data sets](https://www.sec.gov/about/data/edgar-log-file-data-sets); [SRAF EDGAR server log](https://sraf.nd.edu/data/edgar-server-log/)
  - A write-up from when the series ended in 2015 put the raw set at about 1.73 TB uncompressed. A processed 2003-2017 version is available on request (status unverified) — [David Park data page](https://sites.google.com/view/davidpark/data)
- **FINRA equity short interest:** FINRA keeps one rolling year online, with archived files available for download. The archive splits files before 3 Nov 2018 from Nov 2018-Jul 2019, but I could not confirm the earliest date. A third-party mirror starts on 2018-03-29 and a paid API claims December 2017 — [FINRA short interest data](https://www.finra.org/finra-data/browse-catalog/equity-short-interest/data); [FINRA archives](https://otce.finra.org/otce/EquityShortInterest/staticArchives); [Alphacast mirror](https://www.alphacast.io/datasets/equity-short-interest-39102)
- **SEC comment letters (CORRESP/UPLOAD):** the main paper that uses them for prediction is Ryans (2021, RAST), which classifies comment-letter text (see Q2) — [IDEAS: Ryans 2021](https://ideas.repec.org/a/spr/reaccs/v26y2021i1d10.1007_s11142-020-09565-6.html); [LBS working paper PDF](https://lbsresearch.london.edu/id/eprint/1268/1/comment_letter_text_20191102.pdf)
- **10-K / MD&A text:** Brown, Crowley & Elliott (2020) used 10-K narratives from 1994-2012 — [Gies news release](https://giesbusiness.illinois.edu/news/2019/11/25/topic-analysis-a-strong-predictor-of-financial-misreporting); [SMU repository](https://ink.library.smu.edu.sg/soa_research/1828). Bhattacharya & Mickovic (2024) used MD&A sections of 10-Ks taken from the SEC database, reportedly for 1994-2013 (per secondary sources) — [UvA summary](https://www.uva.nl/shared-content/subsites/amsterdam-business-school/en/sections/accounting/amsterdam-accounting-impact-centre-aaic/research-insights/can-context-in-financial-reports-help-detect-fraud.html); [IDEAS](https://ideas.repec.org/a/eee/ijoais/v53y2024ics1467089524000150.html)
- **XBRL tag counts and extensions:** Hoitash & Hoitash (TAR 2018) built an accounting reporting complexity measure from the count of XBRL tags in 10-Ks — [Financial Executives summary](https://daily.financialexecutives.org/measuring-accounting-disclosure-complexity-xbrl/); [Compliance Week](https://complianceweek.com/as-complexity-rises-quality-slides-xbrl-based-study-says/13335.article)
- **Benford / first digits:** the FSD score of Amiram, Bozanic & Rouen (RAST 2015) uses the first-digit distribution of all the numbers in a firm's financial statements. Their sample covers US public firms from 2001-2011 — [IDEAS](https://ideas.repec.org/a/spr/reaccs/v20y2015i4d10.1007_s11142-015-9333-z.html); [Harvard CorpGov forum](https://corpgov.law.harvard.edu/2014/07/09/a-new-tool-to-detect-financial-reporting-irregularities)

### Inferences
- Linkage: insider data (issuer CIK), comment letters, 8-Ks, EDGAR logs and XBRL are all keyed by CIK, so the farr GVKEY-CIK table covers them at the same roughly 87% rate as the existing EDGAR features. The SEC page I fetched did not name the CIK field in the insider files; the table PDF needs checking.
- Time coverage against the 1991-2014 window:
  - Text: about 1994/1996-2014. Best coverage.
  - Insider XML: 2003-2014 from raw EDGAR, 2006-2014 from the SEC flat files.
  - EDGAR logs: 2003-2014.
  - Comment letters: about 2005-2014 (unverified start).
  - 8-K item codes: about Aug 2004-2014 (unverified).
  - XBRL: about 2009/2011-2014 (unverified phase-in). The project already has XBRL companyfacts for 2010-2021.
  - FINRA short interest: none in the training window. Free FINRA data seems to start around 2017-2018 at the earliest, so it only helps a 2018-2020 extension.
- Free price data (Stooq/Yahoo) for returns was not researched in this budget. Since market-value growth and volatility features from Compustat price already work, the expected gain from daily-return features is smaller than from the new families above.

### Gaps
- Exact start years for public comment-letter release (believed to be filings after Aug 2004, released from 2005), 8-K item codes (believed Aug 2004), mandatory electronic Form 4 filing (believed June 2003) and XBRL phase-in (believed 2009-2011). These come from background knowledge, not sources fetched here.
- The earliest FINRA archive file date was not confirmed.
- Not researched: DEF 14A CEO characteristics, related-party disclosures, segment data and the SEC Financial Statement Data Sets page.

## Q2. Which recent papers show incremental out-of-sample value over financial-statement models, and how big is it?

### Takeaway
The strongest evidence is for 10-K/MD&A content features: topics (Brown et al. 2020, up to +59% detection gain) and contextual embeddings (Bhattacharya & Mickovic 2024, +12-15%). These are different from the LM tone and summary features the project already rejected. Evidence for insider trading, comment letters, 8-K events and Benford is either about restatements or material weaknesses rather than AAERs, or is contested.

### Cited Findings
- **Topic models (strongest published evidence).** Brown, Crowley & Elliott (JAR 58(1), 2020) apply a Bayesian topic model to 10-Ks from 1994-2012. Labels are AAERs plus restatement and amendment irregularities. They report topic "significantly improves" out-of-sample detection, by up to 59%, when added to models built on financial and textual style variables — [SMU repository](https://ink.library.smu.edu.sg/soa_research/1828); [EconPapers](https://econpapers.repec.org/RePEc:bla:joares:v:58:y:2020:i:1:p:237-291)
  - Misreporting firms discuss performance increases more, and talk less about risk factors, cost commitments and new loans. The authors say topic complements financial metrics rather than replacing them — [Gies news](https://giesbusiness.illinois.edu/news/2019/11/25/topic-analysis-a-strong-predictor-of-financial-misreporting)
- **BERT on MD&A.** Bhattacharya & Mickovic (IJAIS vol. 53, 2024) fine-tune BERT on MD&A and combine it with Compustat features, using AAER labels. They report gains of 15% over text models and 12% over quantitative models (from secondary summaries; not verified in the paper). Their LDA comparison used 78 topics chosen on validation AUC — [UvA summary](https://abs.uva.nl/content/sections/accounting/amsterdam-accounting-impact-centre-aaic/research-insights/can-context-in-financial-reports-help-detect-fraud.html); [IDEAS](https://ideas.repec.org/a/eee/ijoais/v53y2024ics1467089524000150.html)
- **BERT replication.** A follow-up covering 1994-2021 with AAER labels and a 23-year test period reports average AUCs of 0.844, 0.661 and 0.797 for 2014-2021 across BERT variants. The spread between variants is large. This is a search summary of a ResearchGate item that I did not open — [ResearchGate: AI in fraud detection, textual analysis of 10-K](https://www.researchgate.net/publication/392594631_Artificial_Intelligence_in_fraud_detection_textual_analysis_of_10-K_filings)
- **Multi-agent LLM.** FraudDebate-Agent (Mathematics, MDPI) reports out-of-time per-year test AUCs of 0.78-0.83 that beat RUSBoost every year, with an AAER positive rate of about 0.7%. This is a lower-tier venue and I only saw a summary — [DOI 10.3390/math14152695](https://doi.org/10.3390/math14152695)
- **Abnormal disclosure.** Hoberg & Lewis (JCF 2017, background) find fraudulent firms' MD&A is abnormal against industry, size and age peers. The abnormal text predicts fraud out of sample and is not explained by peer disclosure or firm fixed effects — [IDEAS](https://ideas.repec.org/a/eee/corfin/v43y2017icp58-85.html)
  - Fraud firms under-discuss liquidity, and expense-fraud firms over-discuss R&D — [CFA digest](https://rpc.cfainstitute.org/research/cfa-digest/2017/09/dig-v47-n9-4)
- **Audit and market variables.** Bertomeu, Cheynel, Floyd & Pan (RAST 2021) use gradient-boosted trees on accounting, capital-market, governance and audit variables. Accounting variables "perform poorly alone" but become important through interactions with audit and market variables. They also test one- and two-year-ahead prediction — [IDEAS](https://ideas.repec.org/a/spr/reaccs/v26y2021i2d10.1007_s11142-020-09563-8.html); [Springer PDF](https://link.springer.com/content/pdf/10.1007/s11142-020-09563-8.pdf)
- **Comment letters and EDGAR logs.** In Ryans (RAST 2021), a Naive Bayes classifier on comment-letter text identifies letters linked to future restatements and write-downs. An investor-attention measure built from EDGAR download logs also predicts these outcomes. The outcome is restatements, not AAERs — [IDEAS](https://ideas.repec.org/a/spr/reaccs/v26y2021i1d10.1007_s11142-020-09565-6.html); [LBS](https://lbsresearch.london.edu/id/eprint/1268)
  - Firms more likely to get a 10-K comment letter tend to be less profitable, more complex, use small auditors and have weaker governance (Cassell, Dreher & Myers 2013, background) — [LBS PDF reference list](https://lbsresearch.london.edu/id/eprint/1268/1/comment_letter_text_20191102.pdf)
- **Benford (contested).** The FSD score is higher for misstating firms and rises in the two years before AAERs. Restated numbers conform more closely to Benford than the original misstated ones — [Harvard CorpGov](https://corpgov.law.harvard.edu/2014/07/09/a-new-tool-to-detect-financial-reporting-irregularities)
  - Hopkins (Econ Journal Watch) argues FSD does not beat the F-score or a simple sales-growth screen as a predictor or leading indicator — [Econ Journal Watch](https://econjwatch.org/articles/compared-to-what-does-benford-s-law-really-detect-corporate-fraud)
  - A 2025 arXiv paper argues the 84-86% "conformity" figure reflects low test power — [arXiv 2506.09915](https://arxiv.org/pdf/2506.09915)
- **XBRL complexity.** The number of XBRL tags and extensions is linked to a higher chance of restatements and material-weakness disclosures (Hoitash & Hoitash 2018). This is association, not out-of-sample AAER prediction — [Financial Executives](https://daily.financialexecutives.org/measuring-accounting-disclosure-complexity-xbrl/)
  - A CUNY dissertation (2015-2017 data) finds custom-tag use relates to lower reporting quality — [CUNY Academic Works](https://academicworks.cuny.edu/hc_sas_etds/464)
- **Auditor and CFO turnover (practitioner evidence only).** In an Audit Analytics model, a "critical" auditor change (with disputes) raises material-weakness probability by 12.74 points, and a critical CFO change by 3.65 points. No baseline comparison is given — [Audit Analytics blog](https://blog.auditanalytics.com/predicting-material-weakness/)
  - An unpublished study of about 16,000 auditor dismissals (2000-2013) reports that dismissals outside the usual renewal window roughly double restatement odds over two years — [CFO.com](https://www.cfo.com/news/fire-auditors-early-cfos-are-advised/660169/)
- **Insider trading.** Bao et al. (2020) deliberately use only financial data and acknowledge that useful nonfinancial predictors exist — [CLS Blue Sky](https://clsbluesky.law.columbia.edu/2020/06/05/introducing-machine-learning-to-corporate-fraud-detection/)
  - Agrawal & Cooper (2015, JCF), "Insider trading before accounting scandals", is the standard reference, but I did not verify its findings here — [IDEAS reference list for Bao et al.](https://ideas.repec.org/a/bla/joares/v58y2020i1p199-235.html)

### Inferences
- Effect sizes are not comparable across papers: labels (AAER vs. restatement), test windows and baselines all differ. The +59% and +15% figures are relative detection gains, not AUC points.
- Against the project's +0.002 validation-AUC gate, topic or embedding content features are the only family with published AAER-specific, out-of-sample incremental evidence over financial models, using 10-K text that covers the 1994-2014 window.
- LM tone and "summary" text features already failed here. That fits Brown et al.'s point that content (topic), not style or tone, carries the extra signal.

### Gaps
- I found no peer-reviewed 2019-2026 paper that tests insider-trading features as an incremental AAER predictor with out-of-time evaluation.
- I found none that adds 8-K turnover or auditor-change indicators to a financial-statement fraud model and reports out-of-sample AUC.
- The exact AUC tables of Brown et al. and Bhattacharya & Mickovic were not retrieved.

## Q3. Which signals risk reflecting detection rather than prediction, and what is the missingness trap?

### Takeaway
Several "free" signals are partly by-products of detection: 8-K 4.02 non-reliance, comment letters, CFO exits around restatements, and EDGAR attention. On top of that, every source that starts in 2003-2009 creates an era-and-linkage missingness pattern like the known EDGAR-link flag trap (flag-only AUC about 0.57).

### Cited Findings
- CFO exits cluster around restatements: the chance of a CFO leaving in the window from 3 months before to 9 months after a restatement is 19-29%. Turnover is therefore often a consequence of discovery — [CFO.com](https://www.cfo.com/news/cfos-exit-restating-companies-study-finds/668487/)
- Restating firms with CEO turnover are more likely to receive a later AAER, so turnover after discovery forecasts enforcement rather than misreporting — [Emerald](https://www.emeraldinsight.com/insight/content/doi/10.1108/01140581011091666/full/html)
- Given a comment letter, review teams detect a restatement-causing error in 4.6% of cases, while firms eventually restate 13.6% of reviewed periods. Comment letters are partly part of the detection process (attribution of this working paper is uncertain) — [scite author page](https://scite.ai/authors/matthew-kubic-wVmrlL)
- Restated numbers conform more closely to Benford, so FSD must be computed on the originally reported figures, not restated ones — [Harvard CorpGov](https://corpgov.law.harvard.edu/2014/07/09/a-new-tool-to-detect-financial-reporting-irregularities)

### Inferences
- **Look-ahead rules:**
  - 8-K 4.02 non-reliance is a restatement announcement. It belongs on the label side, not among the features.
  - Use comment letters only after their public release date (released with a delay after review closes), counted for fiscal year t only if released before the year-t 10-K date.
  - EDGAR-log attention should be measured before the fiscal year-end or the filing date.
- **Compustat values:** Bao et al.'s Compustat values may already be restated (Compustat backfills restated values in some fields). So Benford on these 28 items is both low-powered (only 28 numbers per firm-year) and possibly contaminated. FSD needs the full as-filed statements, for example from XBRL 2010+ or the SEC Financial Statement Data Sets.
- **Missingness controls** for any EDGAR-era source:
  - Run a "has-source" flag-only AUC check, as the project did for the EDGAR link.
  - Train and evaluate only on the linked subsample, or on era-restricted folds (for example 2003+ for insider features), and compare to the baseline on that same subsample.
  - Encode "no insider trades" separately from "no data".

### Gaps
- I did not find documented release lags for comment letters or published guidance on point-in-time timestamps for the Insider Transactions Data Sets (filing date vs. transaction date). Both need checking before use.

## Q4. Given what already failed, which untested free families most plausibly add information?

### Takeaway
Ranked by expected value per effort:
1. 10-K/MD&A content features (topics or FinBERT/LLM embeddings, plus abnormal-versus-peer disclosure).
2. Insider-selling features for 2003/2006-2014.
3. EDGAR-log attention and comment-letter text for 2003/2005-2014.
4. XBRL extension and complexity features (2010+ only).

8-K turnover, Benford on 28 items, and FINRA short interest are low priority for the 1991-2014 window.

### Cited Findings
- Topic content adds up to 59% detection over financial and style models in 1994-2012 10-Ks — [SMU](https://ink.library.smu.edu.sg/soa_research/1828)
- Contextual BERT on MD&A adds 12-15% over quantitative and text baselines with AAER labels — [UvA](https://www.uva.nl/shared-content/subsites/amsterdam-business-school/en/sections/accounting/amsterdam-accounting-impact-centre-aaic/research-insights/can-context-in-financial-reports-help-detect-fraud.html)
- Abnormal MD&A relative to industry, size and age peers predicts fraud out of sample — [IDEAS Hoberg-Lewis](https://ideas.repec.org/a/eee/corfin/v43y2017icp58-85.html)
- Accounting variables gain value mainly through interactions with market and audit variables — [IDEAS Bertomeu et al.](https://ideas.repec.org/a/spr/reaccs/v26y2021i2d10.1007_s11142-020-09563-8.html)
- EDGAR attention and comment-letter text predict restatements — [IDEAS Ryans](https://ideas.repec.org/a/spr/reaccs/v26y2021i1d10.1007_s11142-020-09565-6.html)
- XBRL tag and extension counts are associated with restatements and material weaknesses — [Financial Executives](https://daily.financialexecutives.org/measuring-accounting-disclosure-complexity-xbrl/)

### Inferences
- **(1) Text content (top pick).**
  - Why: it is the only family with AAER-specific out-of-sample incremental evidence, and it covers most of the window (about 1994-2014).
  - Feasibility: free from EDGAR. A pre-trained FinBERT or sentence-embedding model on MD&A or Item 7, reduced with PCA, or an LDA with about 50-80 topics fitted only on training years, is doable on a student budget. An abnormal-disclosure variant (distance from the industry/size peer centroid) also reuses the project's peer machinery.
  - Distinctness: it differs from the failed LM tone and summary features.
  - Risks: missing text for unlinked firms creates the link-flag trap (evaluate on the linked subsample), and topic or embedding drift calls for refitting per training window.
- **(2) Insider trading.**
  - Why: it is behavioural and orthogonal to the financials.
  - Feasibility: free flat files only from 2006. Parsing 2003-2005 XML from EDGAR is possible but adds work.
  - Features: net insider selling by officers, CFO/CEO sales, and the share of open-market sales in the year before the 10-K.
  - Limitations: only about 9 of the 24 years are covered. Published AAER-incremental evidence was not found here, so this is a hypothesis to test, not a proven gain.
- **(3) EDGAR-log attention and comment letters (2003-2014).**
  - Why: there is evidence for restatements.
  - Drawbacks: heavy look-ahead and missingness risks, the logs are large, and the outcome studied is restatement rather than AAER.
- **(4) XBRL tag and extension counts (2010+).**
  - Why: cheap to compute from the companyfacts the project already holds.
  - Drawbacks: they only touch the last training/test years and the 2015-2020 extension, so their effect on the main 1991-2014 test AUC will be small or impossible to measure.
- **Lower priority:**
  - Benford on 28 items: too few digits, contested evidence, and possibly restated values.
  - 8-K turnover and auditor changes: often consequences of detection, and only from 2004.
  - FINRA short interest: no free coverage before about 2017-2018.
  - Free daily prices (Stooq/Yahoo): overlaps the market-value features already adopted. Not evaluated here.

### Gaps
- No free-data replication of Bertomeu et al.'s audit variables (auditor identity and fees come from Audit Analytics, which is not free). Whether auditor identity could be parsed for free from 10-K text was not checked.
- DEF 14A CEO characteristics, related-party and segment disclosures, and analyst-data alternatives were not researched within the tool budget.

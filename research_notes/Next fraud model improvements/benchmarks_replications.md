# Benchmarks, replications and critiques for the Bao et al. (JAR 2020) AAER fraud-prediction setting, 2020-2026

Scope note: the numbers below were read from primary PDFs where possible (the Econ Journal Watch exchange, Zavitsanos et al., Alawadhi/Karpoff et al., and the CI-FSFD arXiv paper were text-extracted and read directly). The JAR erratum page returned HTTP 403, so its numbers are taken from (a) the Bao et al. EJW reply table, which matches what the erratum reports per Walker (2022), and (b) Walker's quotations of the erratum. Leakage flags: **[LEAK: random split]**, **[LEAK: no gap]**, **[LEAK: no serial-fraud handling]**, **[LEAK: firm overlap]**.

Reference point for the project: RUSBoost rebuild AUC 0.705 (test 2003-2012), best model 0.748 (expanding window, 2-year gap, serial-fraud masking); 0.788 without masking; 0.828 without gap.

---

## Q1. Replications and critiques of Bao et al. (2020): what AUCs under which rules?

### Takeaway
After the serial-fraud labelling fix (Walker 2021, JAR erratum 2022), RUSBoost on the 28 raw items sits at about **AUC 0.72 for 2003-2008**, about 0.74 for 2003-2005, and about 0.70 for 2006-2008. The originally claimed top-1% advantage (16 hits vs 9 for the Dechow logit) mostly disappears: 8 to 10 hits vs 8 for the 14-ratio logit. Independent Python replications get AUC about 0.715 under strict rules and 0.778 without the serial-fraud fix. That is almost exactly the project's own leak gap (0.748 to 0.788).

### Cited Findings

**Original paper, as later characterised**
- Bao et al. (2020) dataset: 146,045 Compustat firm-years, 1991-2014; 413 unique AAER cases covering 964 fraud firm-years, an unconditional fraud rate of about 0.7%. Rolling design: train 1991 to t-2, test year t, top 1% flagged — [Walker 2021, EJW 18(1)](https://econjwatch.org/File+download/1185/WalkerMar2021.pdf)
- Original published 2003-2008 RUSBoost (28 raw items) results: AUC 0.725, NDCG@k 0.049, sensitivity 4.88%, precision 4.48%. These are secondary figures from a search snippet of a thesis citing the paper, so treat them as approximate. The paper's own Table 3 is the authority — [Bao et al. 2020 via ResearchGate listing](https://www.researchgate.net/publication/337468608_Detecting_Accounting_Fraud_in_Publicly_Traded_US_Firms_Using_a_Machine_Learning_Approach)
- Original claim: RUSBoost "identified a total of 16 fraud cases in the test period 2003-2008", vs 9 for the Dechow et al. model and 7 for Cecchini et al. — quoted in [Walker 2022, EJW 19(2)](https://econjwatch.org/File+download/1245/WalkerSept2022.pdf)
- Serial-fraud rule in the paper (Section 3.3): fraud years in training are recoded to 0 for AAER cases spanning both training and test. Fraud duration: mean 2 years, median 2, 90th percentile 4 — [Walker 2021](https://econjwatch.org/File+download/1185/WalkerMar2021.pdf)
- The released data has 28 raw items plus 14 Dechow-style ratios, and `p_aaer` / `new_p_aaer` identifiers. Tree count is tuned on a 2001 validation year after training on 1991-1999 — [JarFraud/FraudDetection GitHub](https://github.com/JarFraud/FraudDetection)

**Walker (2021) critique, EJW March 2021**
- The bug: `new_p_aaer` split 17 of the 413 AAER cases into two IDs (suffix 1 or 2) when the case was "not consecutive in our sample" because of dropped firm-years with missing Compustat items (e.g., missing DLTIS or XINT). As a result, training-period fraud years of those cases were not recoded to 0, even though the same case appeared in the test year. Example: Delphi, AAER 2504 (2000-2004) — [Walker 2021](https://econjwatch.org/File+download/1185/WalkerMar2021.pdf)
- 10 of the 16 published correct hits (63%) came from these 17 AAERs — [Walker 2021](https://econjwatch.org/File+download/1185/WalkerMar2021.pdf)
- Re-running the authors' MATLAB code with `p_aaer` in place of `new_p_aaer` gave top-1% hits of 16 (published) → 15 (re-run) → **9 (recoded)**. Precision fell 4.5% → 4.2% → **2.5%**, below the 2.63% precision the paper reports for the 14-ratio logit. Per-year recoded hits for 2003-2008: 4, 3, 1, 0, 1, 0 — [Walker 2021](https://econjwatch.org/File+download/1185/WalkerMar2021.pdf)

**Bao et al. (2021) reply, EJW March 2021: results under Walker's rule with tree count re-tuned.** Rule: 28 raw items, train to t-2, serial frauds recoded by p_AAER.

| Test period | RUSBoost-28 raw AUC | Logit-28 raw AUC | Logit-14 ratios AUC | SVM-FK AUC | NDCG@k RUSBoost / Logit-14 | Hits RUSBoost / Logit-14 |
|---|---|---|---|---|---|---|
| 2003-2005 | 0.7428 | 0.6736 | 0.6483 | 0.6255 | 0.0394 / 0.0109 | 9 / 2 |
| 2003-2008 | 0.7228 | 0.6842 | 0.6711 | 0.6195 | 0.0237 / 0.0273 | 10 / 8 |
| 2003-2011 | 0.7171 | 0.6981 | 0.6711 | 0.6297 | 0.0243 / 0.0235 | 13 / 10 |
| 2003-2014 | 0.7196 | 0.7061 | 0.7015 | 0.6519 | 0.0182 / 0.0223 | 13 / 11 |

Source: [Bao, Ke, Li, Yu & Zhang 2021, EJW 18(1), Table 1](https://econjwatch.org/File+download/1184/BaoEtAlMar2021.pdf)

- Under the fix, the logit on 14 ratios beats RUSBoost on NDCG@k for 2003-2008 (0.0273 vs 0.0237) and 2003-2014 (0.0223 vs 0.0182). RUSBoost wins on AUC in every period — [Bao et al. 2021](https://econjwatch.org/File+download/1184/BaoEtAlMar2021.pdf)
- Under Walker's rule, about 20% of fraud firm-years in each test year 2003-2005 are recoded in training, falling to 6.12% for 2006 and 3.74% for 2007 — [Bao et al. 2021, fn. 8](https://econjwatch.org/File+download/1184/BaoEtAlMar2021.pdf)
- Bao et al. argue that 2003-2005 is the cleanest test period, because labels after about 2006 are under-reported (reduced enforcement around the 2008 crisis; Dyck et al. 2010 report about 24 months to initial disclosure). They argue that recoding by p_AAER may be "overkill" when part of a serial fraud was publicly revealed (e.g., by a restatement) before the AAER — [Bao et al. 2021](https://econjwatch.org/File+download/1184/BaoEtAlMar2021.pdf)
- They also note that without any serial-fraud handling (their 2020 Table 6), RUSBoost "would far exceed" the 14-ratio logit — [Bao et al. 2021](https://econjwatch.org/File+download/1184/BaoEtAlMar2021.pdf) **[LEAK: no serial-fraud handling]**

**JAR erratum (2022) and Walker's "Erroneous Erratum" (EJW Sept 2022)**
- The erratum states that Walker "identified an error in the program codes" that "led to an overstatement of model performance metrics". The error left about 10% of spanning serial-fraud cases in training not recoded. The erratum says RUSBoost "continues to dominate" for 2003-2005 — quoted in [Walker 2022](https://econjwatch.org/File+download/1245/WalkerSept2022.pdf); erratum at [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1111/1475-679X.12454) (page not accessible to me, HTTP 403)
- Walker re-ran the erratum code. Grid search over tree counts moves AUC only from 0.7350 to 0.7495, and the optimum is 1,000 trees although the released code uses 300 — [Walker 2022, Table 1](https://econjwatch.org/File+download/1245/WalkerSept2022.pdf)
- Walker's per-year RUSBoost AUC (his two runs): 2003 0.7281/0.7331; 2004 0.7646/0.7670; 2005 0.7289/0.7245; 2006 0.7590/0.7602; 2007 0.7176/0.7184; **2008 0.6280/0.6346** — [Walker 2022, Table 2](https://econjwatch.org/File+download/1245/WalkerSept2022.pdf) (the table layout is garbled in extraction; values are as I aligned them)
- Walker's period summaries:

| Period | AUC (erratum / run 1 / run 2) | NDCG@1% (erratum / run 1 / run 2) | Hits (erratum / run 1 / run 2) |
|---|---|---|---|
| 2003-2005 | 0.7428 / 0.7405 / 0.7415 | 0.0394 / 0.0348 / 0.0357 | 9 / 8 / 8 |
| 2006-2008 | runs ≈ 0.7015 / 0.7044 | 0 | 1 / 0 / 0 |
| 2003-2008 | 0.7228 / 0.7210 / 0.7230 | 0.0237 / 0.0174 / 0.0179 | 10 / 8 / 8 |

  With the released code, Walker gets 8 hits, which equals the 8 hits of the logit benchmark, so the conclusion is "not even a modest improvement" — [Walker 2022](https://econjwatch.org/File+download/1245/WalkerSept2022.pdf)
- Walker (2022) alleges falsification and asks JAR for an investigation. This is his opinion, and Bao et al. had not published a second reply at that time — [Walker 2022](https://econjwatch.org/File+download/1245/WalkerSept2022.pdf). Rejoinder: [Walker Sept 2021](https://econjwatch.org/File+download/1199/WalkerSept2021.pdf)

**Independent open-source replications (not peer-reviewed)**
- Python replication on 2003-2008 (train 1991 to t-2, top 1%, serial fix by AAER id per the erratum, 5 seeds):

| Model | AUC | Top-1% precision | Hits |
|---|---|---|---|
| RUSBoost-28 raw | 0.715 (0.710-0.718) | 1.96% | 7.0 (6-9) |
| LightGBM-28 raw | 0.696 | 2.26% | 8 |
| Logit-14 ratios | 0.691 | 1.98% | 7 |
| Beneish M-score | 0.563 | 1.14% | 4 |
| **RUSBoost without serial fix** | **0.778** | — | 18.4 |

  Random selection gives 0.74% precision — [Teymur102/accounting-fraud-detection](https://github.com/Teymur102/accounting-fraud-detection) **[last row: LEAK, no serial-fraud handling]**
- Second Python replication, with case-level leakage analysis: RUSBoost 8 hits (7-9) under the fixed design. Hits rise to 13.7 without serial-fraud handling and to 19.3 with unrestricted trees. 10 of the 16 published hits come from 20 test-year frauds whose training years were not recoded; these are 8% of test frauds, caught at 50% vs 2.5% for the rest (Fisher p ≈ 3.5×10⁻⁹). **54% of test-year frauds continue cases that began in the training window**, and only **1-3 of 121 genuinely new frauds reach the top 1%**. Logit-14 matches every ML model in the top 1%; RUSBoost wins only on AUC — [ZQ-77777/Bao2020-Fraud-Detection-Replication](https://github.com/ZQ-77777/Bao2020-Fraud-Detection-Replication)

### Inferences
- The project's RUSBoost rebuild (0.705 on 2003-2012) is consistent with the corrected literature (0.7171 for 2003-2011 and 0.7196 for 2003-2014 in the Bao et al. reply; 0.715 for 2003-2008 in an independent Python replication). It is slightly low, plausibly because of later, under-labelled test years (the 2008 AUC is about 0.63).
- **0.748 under the strict rules would be at or above every corrected Bao-dataset number I found** (the best is 0.7428, on the 2003-2005 window that favours RUSBoost). It looks credible but should be presented with the matched test window and a seed or bootstrap interval. Walker's runs show seed and tree-count noise of about ±0.005-0.01 AUC.
- The project's leak gaps match outside evidence. Removing serial masking adds about +0.04 AUC (project 0.748→0.788; Teymur 0.715→0.778, +0.063). Removing the gap adds more (project +0.08). This supports treating any paper without these controls as inflated by roughly 0.05-0.10 AUC.
- AUC and top-1% metrics disagree. After the fix, RUSBoost's AUC edge over the logit survives, but its NDCG@k/top-1% edge does not. The project should report NDCG@k and hits next to AUC, plus a "new-fraud-only" subset. The ZQ analysis suggests that almost all top-1% hits are continuing cases.
- A firm-history Transformer is exactly the kind of model that could profit from continuing-case persistence. Even with label masking, the firm's own earlier feature trajectory is still visible. Report performance on frauds whose case started after the training cutoff.

### Gaps
- I could not open the JAR erratum page (403), so its own tables (including any per-year and NDCG@k figures beyond those quoted by Walker and in the EJW reply) were not read directly.
- The original 2020 Table 3 numbers (0.725 / 0.049) come from a secondary snippet, not the paper PDF.
- I did not find a published response from Bao et al. to Walker (2022), or any JAR action after 2022.

---

## Q2. Papers 2020-2026 using the Bao dataset or AAER labels: best AUC, method, evaluation

### Takeaway
Under strict out-of-time or firm-disjoint evaluation, reported AUCs with AAER-type labels cluster at **0.67-0.75**, whether the model is logit, boosted trees or an LLM. Results above about 0.85 come from random or firm-overlapping splits. The clearest 2026 demonstration (CI-FSFD) shows the same tree models dropping from 0.95-0.96 AUC under random 5-fold CV to 0.66-0.70 under company-isolated CV.

### Cited Findings

**Bertomeu, Cheynel, Floyd & Pan (2021), Review of Accounting Studies 26:468-519**
- Uses GBRT on a broad set of accounting, capital-market, governance and audit variables to detect material misstatements (restatement-based) and AAERs. Its key finding is that accounting variables are weak alone but become useful when interacted with audit and market variables — [Semantic Scholar TLDR / record](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1007/s11142-020-09563-8?fields=title,abstract,year,venue,tldr); [WUSTL profile](https://profiles.wustl.edu/en/publications/using-machine-learning-to-detect-misstatements/)
- A secondary summary says that on AUC, RUSBoost is somewhat better than GBRT, both beat logit, and the misstatement-trained model predicts AAERs about as well as or better than an AAER-trained model. Not verified against the paper — [search summary; RAST PDF behind paywall](https://link.springer.com/article/10.1007/s11142-020-09563-8)
- A Rotterdam thesis following Bertomeu et al.'s variable set on restatements gets random forest AUC 0.678, rising to 0.696 with extra variables. Its split design was not verified — [Dalebout, EUR thesis](https://thesis.eur.nl/pub/60720/Master-Thesis-Mandy-Dalebout-475530.pdf)

**Alawadhi, Karpoff, Koski & Martin (2023 WP), "The prevalence and price distorting effects of undetected financial misrepresentation"**
- Labels: all SEC and DOJ enforcement actions for financial misrepresentation (actions 1978-2017, violations 1976-2014). Each year 0.27% of Compustat firms start a misrepresentation later sanctioned, and the average violation lasts 3.1 years — [Alawadhi et al. 2023](https://haslam.utk.edu/wp-content/uploads/2023/04/Karpoff-FP-paper-2023.04.23.pdf)
- **k-fold cross-validated AUC is about 0.78** for the parsimonious logit and SVMs (SVC-RBF 78.32%; 79.26% with extra candidate predictors) **[LEAK: random split / firm overlap]**. **Year-by-year rolling out-of-sample AUC is 0.73-0.77** — [Alawadhi et al. 2023](https://haslam.utk.edu/wp-content/uploads/2023/04/Karpoff-FP-paper-2023.04.23.pdf)
- In-sample AUC (1976-2014): Beneish 0.54, Dechow et al. (2011) 0.67. Their full logistic model has in-sample AUC 0.7857; kNN and MLP average under 0.70; the ensemble model's AUC SD is 0.0374, about 7x the logit's — [Alawadhi et al. 2023](https://haslam.utk.edu/wp-content/uploads/2023/04/Karpoff-FP-paper-2023.04.23.pdf)

**Zavitsanos et al. (ICAIF 2021), "Financial misstatement detection: a realistic evaluation"**
- Uses AAER and Audit Analytics restatement labels, financial (Dechow/Bao) and 10-K text features, and SVM/LR evaluated as rankers (ROC-AUC, NDCG, R-precision) on chronological splits — [arXiv 2305.17457](https://arxiv.org/abs/2305.17457)
- **Identity leakage:** an SVM using only the *company name* as feature performed "almost identical" to one using full 10-K text, i.e., text models were memorising serial misstaters — [Zavitsanos et al.](https://arxiv.org/abs/2305.17457)
- Simulating detection delay (training positives flipped to negative if the restatement date falls after the test year) makes performance drop "significantly". About 40% of training positives are unknown at test time: only about 50% of Y-1 positives, 70% of Y-2 and 80% of Y-3 are known. All models then perform similarly poorly, with text, financial or both. Adding a 2-year gap made results worse still. Numbers are shown only in figures — [Zavitsanos et al.](https://arxiv.org/abs/2305.17457)

**Waffo Dzuyo et al. (arXiv July 2026, FinLLM@IJCAI 2026), "Benchmarking Generalization in Financial Statement Fraud Detection" (CI-FSFD)**
- Data: AAER labels (249 AAERs LLM-extracted and expert-verified, fraud years 2000-2022), 10-Q financials 2009-2024 (122 indicators), and LLM-summarised MD&A. Final set 10,159 firm-quarters (511 fraud, downsampled to about 5% fraud). There is no temporal split, which the authors list as future work — [arXiv 2607.19259](https://arxiv.org/html/2607.19259v1)
- Random 5-fold CV **[LEAK: random split, firm overlap]**: XGBoost 0.96, LightGBM 0.95, RF 0.92, MLP 0.89, Llama-3.1-8B (SMD&A) 0.96, LR-Dechow 0.68 — [arXiv 2607.19259](https://arxiv.org/html/2607.19259v1)
- Company-isolated 5-fold CV: XGBoost 0.66, LightGBM 0.68, RF 0.70, MLP 0.69, LR-Dechow 0.67. Best is fine-tuned Fino1-8B on summarised MD&A at **0.74 ± 0.03**; FIN+SMD&A 0.72 — [arXiv 2607.19259](https://arxiv.org/html/2607.19259v1)
- Zero-shot LLMs (Fino1-8B, Llama-3.1-8B, Qwen3-32B) score AUC 0.47-0.52, i.e. chance — [arXiv 2607.19259](https://arxiv.org/html/2607.19259v1)

**Other LLM / text papers**
- Bhattacharya & Mićković (2024, IJAIS 53:100682): BERT fine-tuned on 10-K MD&A, 1994-2013 AAER sample. They report +15% over a textual benchmark and +12% over a quantitative benchmark, and 5x more frauds than the textual benchmark at equal investigation size. No AUC or split details found — [UvA summary](https://abs.uva.nl/content/sections/accounting/amsterdam-accounting-impact-centre-aaic/research-insights/can-context-in-financial-reports-help-detect-fraud.html); [X-MOL record](https://x-mol.com/paper/1769589638189772800)
- Brown, Crowley & Elliott (2020, JAR 58(1):237-291): an LDA topic model on 10-K narratives (1994-2012), with AAER plus restatement/10-K/A irregularity labels. Topic adds significant out-of-sample improvement over financial and style models and helps most for long-duration misstatements. The press release claims "as much as 59%" improvement in some samples; AUC values not retrieved — [IDEAS/RePEc](https://ideas.repec.org/a/bla/joares/v58y2020i1p237-291.html); [Gies press release](https://giesbusiness.illinois.edu/news/2019/11/25/topic-analysis-a-strong-predictor-of-financial-misreporting)
- Tan, Ma & Zhang (ACL 2026, arXiv 2512.13040), FinFRE-RAG: feature reduction plus retrieval-augmented in-context learning on four public tabular fraud datasets. It improves F1/MCC over direct prompting, but "LLMs still trail specialized classifiers". No AUC or temporal-split details in the abstract — [arXiv 2512.13040](https://arxiv.org/abs/2512.13040)
- Ergun & Sefer (2025, ISAFM), "Financial Statement Fraud Detection via Large Language Models": results not retrieved — [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/isaf.70021)

**Ensemble / cost-sensitive ML papers (ESWA / IRFA-type venues)**
- Achakzai & Peng (2023, Int. Rev. Financial Analysis 89) introduce dynamic ensemble selection for financial statement fraud. AUC, labels and split were not retrievable — [IDEAS](https://ideas.repec.org/a/eee/finana/v89y2023ics1057521923003435.html)
- A thesis proposing a new method on this setting reports average AUC 0.693, below RUSBoost; a Chinese-market stacking model reports 0.742 vs RUSBoost about 0.72 on different data (search-summary level only) — [Sustainability 2023, China](https://doi.org/10.3390/su15010105); [Hawaii ScholarSpace thesis](https://scholarspace.manoa.hawaii.edu/server/api/core/bitstreams/e7797e95-2899-4f31-b6d9-5e8a74caafbb/content)

### Inferences
- The realistic ceiling with financial-statement features and AAER-type labels under strict out-of-time or firm-disjoint evaluation is about **0.70-0.75 AUC**, across 2021-2026 work: Bao corrected 0.72-0.74, Python replications 0.70-0.72, Karpoff rolling 0.73-0.77 with a broader enforcement label, and CI-FSFD best 0.74. The project's 0.748 is at the top of this band.
- Any paper reporting more than about 0.85 AUC on AAER labels should be presumed to use a random split or firm overlap unless shown otherwise. CI-FSFD shows such splits add +0.20-0.30 AUC for flexible models but about 0 for a 7-variable logit, so leakage specifically flatters high-capacity models.
- LLMs: zero-shot is at chance, and fine-tuned text models gain only a few AUC points over trees under company isolation. No paper I found evaluates an LLM under a Bao-style temporal split with a gap and serial masking.

### Gaps
- Bertomeu et al. (2021) exact out-of-sample AUCs and split design (paywalled; secondary sources conflict, one citing 0.878, which is unverified and likely a different metric or setting).
- Brown et al. (2020) and Bhattacharya & Mićković (2024) AUC values, and whether their splits had a gap or serial-fraud handling.
- I found no peer-reviewed ESWA/DSS/IEEE deep-learning paper evaluated on the JarFraud split with the erratum fix. Several exist on Chinese CSMAR data or random splits; I did not verify any specific one.

---

## Q3. Which improvements show credible gains under strict temporal evaluation?

### Takeaway
Very few improvements have been shown under the full strict protocol (out-of-time, gap, serial masking). The best-supported are: boosted-tree ensembles over logit on **AUC** (but not on top-1% hits); interactions with **audit and market variables** (Bertomeu); and **text/topic or MD&A** information (Brown et al.; CI-FSFD under company isolation). Gains from hyperparameter tuning, deeper trees, deep networks on raw items, or zero-shot LLMs mostly disappear once leakage is removed.

### Cited Findings
- Ensemble vs logit: under the corrected Bao design, RUSBoost beats the 14-ratio logit on AUC in every window (+0.02 to +0.09). It does not beat it on NDCG@k for 2003-2008 or 2003-2014 — [Bao et al. 2021](https://econjwatch.org/File+download/1184/BaoEtAlMar2021.pdf). Independent replication: LightGBM 0.696 vs RUSBoost 0.715 vs logit 0.691 — [Teymur102](https://github.com/Teymur102/accounting-fraud-detection)
- Tuning gives little gain: tree-count grid search moves AUC within 0.735-0.750 — [Walker 2022](https://econjwatch.org/File+download/1245/WalkerSept2022.pdf). "Under the corrected designs, deeper trees bring no gain"; unrestricted trees help only when serial leakage is present (19.3 hits) — [ZQ-77777](https://github.com/ZQ-77777/Bao2020-Fraud-Detection-Replication)
- Market, audit and governance variables: accounting variables become important only "with suitable interactions" with audit and market variables — [Bertomeu et al. 2021 TLDR](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1007/s11142-020-09563-8?fields=title,abstract,year,venue,tldr). This supports the project's market-feature gain. Extra candidate predictors added only +0.01 AUC (78.32%→79.26%) in Karpoff et al. — [Alawadhi et al. 2023](https://haslam.utk.edu/wp-content/uploads/2023/04/Karpoff-FP-paper-2023.04.23.pdf)
- Text: topic content adds out-of-sample power over financial and style variables, especially for long-duration misstatements — [Brown et al. 2020](https://ideas.repec.org/a/bla/joares/v58y2020i1p237-291.html). Under company isolation, MD&A-summary fine-tuning reaches 0.74 vs 0.66-0.70 for financial-only trees — [CI-FSFD 2026](https://arxiv.org/html/2607.19259v1). But in Zavitsanos et al., text gains vanish once delay is simulated, and raw-text models mostly learn firm identity — [Zavitsanos et al.](https://arxiv.org/abs/2305.17457)
- Combining text and financials did *not* help under company isolation (FIN+SMD&A 0.72 < SMD&A 0.74) — [CI-FSFD](https://arxiv.org/html/2607.19259v1)
- Detection-delay modelling: flipping not-yet-revealed positives to negative (restatement-date-aware labels) is the realistic way to train. It lowers measured performance but is the honest setting. A fixed 2-year gap discards the most recent data and did not improve results in their experiments — [Zavitsanos et al.](https://arxiv.org/abs/2305.17457). Bao et al. similarly argue that labels revealed by earlier events (restatements, litigation) should be usable in training — [Bao et al. 2021](https://econjwatch.org/File+download/1184/BaoEtAlMar2021.pdf)
- Cost-sensitive learning, graph/network and peer features: I found no study showing gains on AAER labels under the strict temporal protocol. Graph-transformer fraud work (e.g., FraudGT) targets transaction fraud, not firm-year misstatements — [safe-graph paper list](https://github.com/safe-graph/graph-fraud-detection-papers)

### Inferences
- The project's market features and ensemble are in line with what has held up (Bertomeu's interaction finding; ensembles beat logit on AUC). The firm-history Transformer is the component most exposed to continuing-case persistence and needs a "new cases only" check.
- Candidate next steps with the best evidence base:
  1. **Delay-aware training labels** (use AAER/restatement reveal dates instead of a blunt 2-year gap) to recover recent training years.
  2. **Text features from 10-K MD&A** (topic or fine-tuned encoder), evaluated firm-disjoint and out-of-time.
  3. **Audit/governance variables** (auditor change, fees, opinions, Big-4) interacted with accounting items via trees.
  4. **Reporting on top-k metrics and new-fraud-only subsets**, since AUC gains have repeatedly failed to translate into more top-1% detections.
- Since the logit-14 ratio model ties ML at the top 1%, a simple rank-average of logit and the ensemble is a cheap test to run.

### Gaps
- No strict-protocol evidence found for cost-sensitive boosting, graph/supply-chain or peer-firm features, or industry-relative features on AAER labels; this remains open.
- No paper found that applies a firm-history sequence model (RNN/Transformer) to the Bao dataset under the erratum rules, so the project's result has no direct comparator.

---

## Q4. Ceiling, main limitations and better labels

### Takeaway
Everyone agrees the binding constraint is the labels: AAERs are rare (about 0.7% of firm-years; 413 cases), revealed with multi-year delay, under-enforced after 2006-2008, and a small, selected subset of actual misreporting. Karpoff et al. estimate that only about 3.5% of firms that misrepresent are ever sanctioned. Broader labels (restatements, class actions, SEC/DOJ actions) raise sample size but change what is being predicted.

### Cited Findings
- Rarity: 964 fraud firm-years out of 146,045 (0.7%) — [Walker 2021](https://econjwatch.org/File+download/1185/WalkerMar2021.pdf). CI-FSFD processed only 249 of 3,300 AAERs into usable labels — [arXiv 2607.19259](https://arxiv.org/html/2607.19259v1)
- Detection delay: about 24 months on average to initial disclosure (Dyck et al. 2010, as cited); labels for test years from about 2006 onward are understated — [Bao et al. 2021](https://econjwatch.org/File+download/1184/BaoEtAlMar2021.pdf). At a 2003 test date, only about 50% / 70% / 80% of Y-1 / Y-2 / Y-3 restatement positives are known — [Zavitsanos et al.](https://arxiv.org/abs/2305.17457)
- Enforcement regime shift: Bao et al. ended the main test at 2008 because of a "noticeable shift" in enforcement around the financial crisis — quoted in [Walker 2022](https://econjwatch.org/File+download/1245/WalkerSept2022.pdf). Per-year RUSBoost AUC falls to about 0.63 in 2008 — [Walker 2022](https://econjwatch.org/File+download/1245/WalkerSept2022.pdf)
- Undetected fraud: the base logistic model implies 22.3% of Compustat firms engage in potentially sanctionable misrepresentation in an average year; 7.2% initiate each year; only 3.5% of these are eventually sanctioned — [Alawadhi et al. 2023](https://haslam.utk.edu/wp-content/uploads/2023/04/Karpoff-FP-paper-2023.04.23.pdf). This is a model-based estimate and an author claim.
- Serial-fraud handling has no consensus: Walker recodes by p_AAER; Bao et al. argue for reveal-date-based handling — [Bao et al. 2021](https://econjwatch.org/File+download/1184/BaoEtAlMar2021.pdf)
- Better or alternative labels:
  - Audit Analytics restatements (Zavitsanos; Bertomeu "misstatements")
  - AAER plus restatement and 10-K/A irregularities (Brown et al.)
  - The full SEC+DOJ enforcement database (Karpoff/Alawadhi), which achieves rolling AUC 0.73-0.77

  Sources: [Zavitsanos](https://arxiv.org/abs/2305.17457); [Brown et al.](https://ideas.repec.org/a/bla/joares/v58y2020i1p237-291.html); [Alawadhi et al.](https://haslam.utk.edu/wp-content/uploads/2023/04/Karpoff-FP-paper-2023.04.23.pdf). Restatement-trained models predict AAERs about as well as AAER-trained ones (secondary summary of [Bertomeu et al.](https://link.springer.com/article/10.1007/s11142-020-09563-8))
- Restatement samples are "harder", since they may include unintentional errors — [Zavitsanos et al.](https://arxiv.org/abs/2305.17457)

### Inferences
- With about 10 to 20 positive test firm-years per year, single-period AUC differences under about 0.02-0.03 are within seed and window noise (Walker's runs; the replication seed ranges). The project's +0.043 over its own RUSBoost (0.705→0.748) is larger than that, but should come with bootstrap CIs.
- A label expansion (AAER ∪ AA restatements with fraud/irregularity flags ∪ Stanford SCAC class actions) is the most commonly endorsed route to more positives. It must use reveal dates to avoid look-ahead, and evaluation should stay on AAER labels for comparability.

### Gaps
- I did not retrieve a study that quantifies the AUC gain from combining AAER + restatements + class actions under a strict temporal split. Stanford SCAC and SEC litigation-release label combinations were not directly evaluated in sources I could read.
- The Karpoff 22.3% prevalence estimate depends on the authors' bivariate/logit modelling and has not been independently replicated in sources found.

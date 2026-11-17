# Modelling and learning methods for AAER-labelled accounting-fraud detection (2019-2026) with out-of-time evidence

Scope note: the search budget was about 16 tool calls. Several full texts were paywalled (Springer RAST, Wiley Accounting & Finance) or only readable as abstracts or secondary summaries. These cases are flagged below. "Strict OOT" means a chronological train/test split. Where a paper used a gap or serial-fraud handling, that is stated explicitly.

## 1. Label-noise and detection-delay methods (censoring, delay-aware labels, PU learning, cleaning negatives)

### Takeaway
The only AAER/restatement study found that models detection delay explicitly under a temporal split is Zavitsanos et al. (arXiv 2305.17457). They relabel training positives "as known at test time" and find that realistic delay makes every model much worse. Neither a gap nor shorter windows recovered performance. No study found reports an out-of-time AUC gain from PU learning, co-teaching, survival/censoring framing or negative-cleaning on US AAER data. The positive results for noise-robust methods come from Chinese data, graph models, or unspecified splits.

### Cited Findings
- **Zavitsanos et al., "Financial misstatement detection: a realistic evaluation" (2023).**
  - Data: US 10-K filings 1998-2008, using the Bao et al. financial data truncated at 2008. There are 4 datasets of 32,443 samples each with about 1% positives. Labels are AAER (mostly intentional) or Audit Analytics (AA) restatements (which include errors). Features are 42 (Bao's 28 plus 14 Dechow-derived) or MD&A TF-IDF. — [arXiv 2305.17457 (HTML)](https://arxiv.org/html/2305.17457v1)
  - Evaluation: a sliding 3-year training window tested on the next year (e.g. train 2000-2002, test 2003), with 5-fold CV for tuning inside each window. Random shuffling is rejected because it leaks future information. — [arXiv 2305.17457](https://arxiv.org/html/2305.17457v1)
  - **Detection-delay simulation ("AA-Hard"):** for each test year, a training positive whose restatement date falls after that test year is relabeled negative.
    - About 40% of training positives flip.
    - For test year 2003, only about 50% of Y-1, 70% of Y-2 and 80% of Y-3 positives were already known.
    - Under this realistic labelling, all models (SVM, LR) with text, financial or combined features "perform similarly poorly".
    - Dropping the last two years (Bao's gap) made performance drop further. 1- and 2-year gaps with 3- or 5-year windows gave no improvement.
    - Results are reported as R-precision in figures, not as AUC/NDCG. — [arXiv 2305.17457](https://arxiv.org/html/2305.17457v1)
  - Text-model warning: using only the company name gave almost the same performance as the full text. The authors read this as overfitting to firm identity, which is relevant to serial-fraud leakage. Feature selection reduced performance. — [arXiv 2305.17457](https://arxiv.org/html/2305.17457v1)
  - The authors say unknown positives among the negatives motivate PU learning, but they do not implement it. — [arXiv 2305.17457](https://arxiv.org/html/2305.17457v1)
  - The abstract states that "the evaluation process significantly affects system performance". — [arXiv abs 2305.17457](https://arxiv.org/abs/2305.17457)
- **Zavitsanos & Paliouras, "Focused PU learning from imbalanced data" (Data Mining and Knowledge Discovery 40, 101, 2026; arXiv 2605.14467).**
  - Method: a "focused empirical risk estimator" for hard positives that resemble negatives. It is evaluated under SCAR and SAR labelling mechanisms, with financial misstatement detection as a real-world application.
  - The abstract page does not say whether the misstatement experiment is temporal, and gives no numbers. — [arXiv 2605.14467](https://arxiv.org/abs/2605.14467)
  - Note: this appears to be the same research line as the iFPU focal-PU risk that the project already tested and rejected.
- **Hidden fraud on a Chinese corporate graph (arXiv 2502.19305, 2025).**
  - Treats "hidden fraud" (undetected violations) as noisy training labels. The method, KeGCN_R, uses two-stage learning for robustness.
  - Data: 18 years of Chinese financial data in three graph datasets.
  - The abstract gives no gain sizes or split design. — [arXiv 2502.19305](https://arxiv.org/abs/2502.19305)
- **PU learning on China A-share accounting fraud (Xi'an Jiaotong).** The summary claims PU learning enhances detection accuracy and recall. AUC figures and split design were not visible. — [XJTU Scholar record](https://scholar.xjtu.edu.cn/en/publications/detecting-accounting-fraud-in-china-a-share-market-with-pu-learni/)
- **Generic PU-for-extreme-imbalance work (arXiv 2004.09820).**
  - On credit-card fraud data, it reports AUC of about 0.97 (ProbTagging) and about 0.89 (nnPU).
  - Performance degrades as observed positives get fewer.
  - This is not a financial-statement setting and not a temporal split. — [arXiv 2004.09820](https://arxiv.org/pdf/2004.09820)
- **Bertomeu et al. (RAST 2021): earliness of detection.** Of the 29 misstatements caught by GBRT in the test sample, 20 relate to AAER filings, and all were detected at least a year before the AAER was filed. — [Springer RAST article](https://link.springer.com/article/10.1007/s11142-020-09563-8) (search snippet; full text paywalled)
- **Walker critique of Bao et al. (Econ Journal Watch, March 2021).**
  - Section 3.3 covers serial fraud.
  - The critique reports that AAER identifiers changed between the training and test sets. As one example, if Delphi's AAER identifier had not been changed, the model would not have flagged Delphi's 2003 fraud.
  - This illustrates how AAER/serial-fraud label construction can drive measured out-of-time AUC. — [Walker 2021, EJW](https://econjwatch.org/File+download/1185/WalkerMar2021.pdf)

### Inferences
- "Delay-aware relabelling" (training labels as known at the test date) is mainly an evaluation-realism device. The only temporal evidence shows that it *lowers* measured performance. It is not shown to raise AUC.
  - The student's 2-year gap is already a partial version of this. Zavitsanos et al. found that gaps did not help *their* linear models, which is consistent with the gap being a fairness constraint rather than a performance lever.
- One cheap, untested idea with a new angle is **delay-aware sample weighting:** down-weight negatives in the last 1-3 training years by the probability that a fraud there is still undetected.
  - That probability can be estimated empirically from the AAER date minus the fiscal-year lag in the student's own label file. Zavitsanos et al.'s known-fraction figures (about 50/70/80% at 1/2/3 years) give a rough prior.
  - This differs from the covariate-shift weights and recency weights already tested. However, no published OOT AUC gain was found for it, so it is speculative and should go through the +0.002 gate.
- Co-teaching, small-loss negative cleaning and survival/censoring framings: no AAER-specific out-of-time evidence was found. Given the failure of iFPU (PU risk) in this project, the expected value is low.
  - If tried, the cheapest version is a "confident-learning" pass: drop the top-k% highest-scored negatives from training folds only, then retrain. It is laptop-feasible.

### Gaps
- No 2019-2026 paper was found that reports an out-of-time AUC/NDCG *gain* on US AAER data from PU learning, co-teaching, confident learning, survival/hazard models, or delay-aware weighting.
- The numeric R-precision values in Zavitsanos et al. are only in figures and could not be extracted.
- The focused-PU paper's misstatement-experiment design (temporal or random) and its numbers are unknown from the abstract.
- Dyck, Morse & Zingales-style estimates of the undetected-fraud share (useful to set PU class priors) were not retrieved in this session.

## 2. Tabular foundation models (TabPFN v2 / 2.5, TabICL, etc.) at about 100k-150k rows and about 1% positives

### Takeaway
On the only large, highly imbalanced financial-distress benchmark found (about 1M firm-years, 0.2-0.4% bankruptcies), TabPFN *lost* to XGBoost, CatBoost and LightGBM at every horizon. The gap widened at longer horizons (e.g. AUC 0.771 vs 0.891 at h=4), at roughly 3,000x the compute. Vendor benchmarks claim TabPFN-2.5 beats tuned GBDTs, but these are mostly on small and medium, non-temporal datasets. The most plausible use for this project is as a **diverse ensemble member trained on a class-balanced context subsample**, not as a replacement.

### Cited Findings
- **TabPFN-2.5 (Prior Labs, arXiv 2511.08667, Nov 2025).**
  - Designed for up to 50,000 rows, though the authors say the limit is not strict. They report strong results with up to 100,000 training samples. Prior Labs docs list 100,000 rows by 2,000 features for TabPFN-2.5. — [arXiv 2511.08667](https://arxiv.org/html/2511.08667v2); [Prior Labs docs mirror](https://gh.r-mj.com/PriorLabs/docs/blob/main/models.mdx)
  - Claims to outperform tuned XGBoost and CatBoost on TabArena and to match AutoGluon 1.4 tuned for 4 hours. It reports a 100% default-vs-default win rate against XGBoost on small and medium classification tasks. — [arXiv 2511.08667](https://arxiv.org/html/2511.08667v1)
  - **License:** the weights are under the "TABPFN-2.5 License v1.0", which is non-commercial. The code is under the Prior Labs License (Apache 2.0 plus attribution). Academic and student use is therefore allowed, but commercial use needs a separate agreement. — [arXiv 2511.08667](https://arxiv.org/pdf/2511.08667v2); [PriorLabs/TabPFN GitHub mirror](https://gh.r-mj.com/PriorLabs/TabPFN)
  - Prior Labs' "Scaling Mode" claims to remove the row limit, with strong results up to 10M rows. This is a vendor technical report. — [Prior Labs technical report](https://priorlabs.ai/technical-reports/large-data-model)
  - The docs also list later TabPFN-2.6 and TabPFN-3 releases. — [Prior Labs docs mirror](https://gh.r-mj.com/PriorLabs/docs/blob/main/models.mdx)
- **Kostrzewa et al., "Are Foundation Models Useful for Bankruptcy Prediction?" (NeurIPS 2025 workshop; arXiv 2511.16375).**
  - Data: 1,106,879 firm-years from Poland, Hungary, Slovakia and Czechia, 2006-2021. Positive rates are about 0.36% (h=0) down to about 0.21% (h=4).
  - Split: stratified test subset of 20,000. **The paper does not state that the split is temporal.**
  - Because TabPFN is described as designed for fewer than 10,000 samples, the authors used **TabPFN-DT**: a shallow tree with at least 10k samples per leaf, with a TabPFN model in each leaf. — [arXiv 2511.16375 (HTML)](https://arxiv.org/html/2511.16375v1)

  | Horizon | XGBoost AUC | TabPFN AUC |
  |---|---|---|
  | h=0 | 0.996 | 0.987 |
  | h=1 | 0.968 | 0.951 |
  | h=2 | 0.894 | 0.800 |
  | h=3 | 0.896 | 0.823 |
  | h=4 | 0.891 | 0.771 |

  CatBoost and LightGBM were similar to XGBoost. — [arXiv 2511.16375](https://arxiv.org/html/2511.16375v1)
  - Bootstrap TabPFN ensembles reached 0.794-0.814 AUC at h=4, still below the GBDTs. — [arXiv 2511.16375](https://arxiv.org/html/2511.16375v1)
  - Compute: TabPFN on an A100-40GB took about 23.8 s per 20k-sample batch. XGBoost on a 4-thread CPU took about 0.007 s. The authors say TabPFN's cost "cannot be justified by its performance". — [arXiv 2511.16375](https://arxiv.org/html/2511.16375v1); [NeurIPS 2025 page](https://neurips.cc/virtual/2025/132483)
- **Tanna et al., "Data Presentation Over Architecture" (arXiv 2605.18635, 2026). Credit default on Home Credit and Lending Club.**
  - Compares RF, XGBoost, LightGBM and CatBoost with TabPFN v2.5, TabICL, OrionMSP and OrionBix, using context sizes from 1K to 50K.
  - How the context is built explains more AUC variance than which foundation model is used. Balanced or hybrid context sampling adds 3-4 AUC points over uniform sampling.
  - Per secondary summaries, a balanced 5K-10K context lets TabPFN and TabICL reach AUC comparable to GBDTs trained on the full data, i.e. comparable rather than better. The temporal split design was not verified. — [arXiv 2605.18635](https://arxiv.org/pdf/2605.18635); [Moonlight review](https://www.themoonlight.io/en/review/data-presentation-over-architecture-resampling-strategies-for-credit-risk-prediction-with-tabular-foundation-models)

### Inferences
- At about 100k-150k training rows with about 1% positives (about 1,000-1,500 frauds), TabPFN-2.5 is at or above its stated comfort zone. The bankruptcy evidence suggests GBDTs will win as a standalone model.
  - A cheap, laptop-feasible variant matches the student's existing "balanced bags" design: feed TabPFN or TabICL a *balanced context* of all training positives plus about 5-10k sampled negatives, bag several such contexts, and rank-average into the ensemble.
  - The value would come from diversity, since in-context learning makes errors uncorrelated with trees. No fraud-specific evidence of a gain exists, so it must pass the +0.002 gate.
- Practical feasibility: TabPFN can run on CPU for small contexts but is slow. A free Colab or Kaggle GPU is the realistic route. The non-commercial weights license is fine for a student project. TabICL is reported to target larger tables, but this was not verified here.
- Test-year scoring cost scales with the number of test rows times the context size. With about 10k test firms per year across 10 test years, inference is feasible on a free GPU.

### Gaps
- No paper was found that applies TabPFN, TabICL or CARTE to AAER or US financial-statement fraud data.
- No foundation-model study found used a strict temporal split with a gap on financial distress data. Kostrzewa et al. used a stratified split.
- TabICL's row limits and licensing, and CARTE's applicability (it targets string-valued, heterogeneous tables, which is less relevant to numeric Compustat), were not verified in this session.

## 3. Multi-task / auxiliary-target learning, transfer and semi-supervised learning

### Takeaway
No accounting study was found that trains AAER fraud with auxiliary targets (restatements, going-concern opinions, comment letters) in a multi-task setup, and no out-of-time evidence of such gains exists in this literature. The closest evidence is about **label choice**. AAER positives are more separable than Audit Analytics restatements. Restatement detection is hard and "ensemble methods also struggle". Restatement labels are noisier, so their value as auxiliary signals is uncertain.

### Cited Findings
- AAER labels (mostly intentional misstatements) make the task easier than AA restatement labels, because the classes are more distinguishable. AA records the restatement date for each misstatement, which enables delay-aware labelling. — [Zavitsanos et al. 2023](https://arxiv.org/html/2305.17457v1)
- Bertomeu et al. (RAST 2021):
  - Accounting variables alone detect misstatements poorly but become informative when interacted with **audit and market variables**.
  - The paper examines one- and two-year-ahead prediction and contrasts misstatements with irregularities. — [IDEAS/RePEc record](https://ideas.repec.org/a/spr/reaccs/v26y2021i2d10.1007_s11142-020-09563-8.html); [Springer](https://link.springer.com/article/10.1007/s11142-020-09563-8); [WUSTL profile](https://profiles.wustl.edu/en/publications/using-machine-learning-to-detect-misstatements/)
- Evaluations of ML for predicting restatements find restatement detection "remains a difficult task, with ensemble methods also struggling". — [ResearchGate: Evaluating ML Algorithms in Predicting Financial Restatements](https://www.researchgate.net/publication/347278311_Evaluating_Machine_Learning_Algorithms_in_Predicting_Financial_Restatements)
- An ACM Computing Surveys review (2025) on ML for identifying risk in financial statements is a starting point for multi-task and transfer studies. Its specific multi-task findings were not extracted. — [ACM CSUR survey](https://dl.acm.org/doi/full/10.1145/3723157)

### Inferences
- Bertomeu et al.'s result suggests the more productive "auxiliary" direction is **auxiliary inputs** (audit fees, auditor changes, going-concern opinions, Big-N, market variables) rather than auxiliary targets. The project already adopted market-value features. Audit-variable inputs (from Audit Analytics, typically via WRDS) are not in its "tested" list.
  - Feasibility depends on whether the student has WRDS/Audit Analytics access. It is not free.
- If the student has AA restatements, a feasible multi-task variant is to train LightGBM on the AAER target with a restatement-probability feature produced *out-of-fold within the training window*, or to add a multi-head output to the Temporal Transformer.
  - The restatement labels must also respect delay (use the restatement announcement date). Expected gains are unknown, and the label noise documented above is a risk.
- Semi-supervised and transfer learning: no AAER out-of-time evidence was found. Self-training on unlabeled firm-years is close in spirit to the PU approaches already tested and rejected, so its prior is low.

### Gaps
- No 2019-2026 paper found evaluates multi-task learning with restatements, going-concern opinions or SEC comment letters as auxiliary targets for AAER detection.
- No paper found evaluates transfer learning (e.g. pretrain on restatements, fine-tune on AAER) or semi-supervised learning under out-of-time splits.
- Brown, Crowley & Elliott (JAR 2020, topic-model text features) was not retrieved in this session. It is text-based, so it is out of scope for numeric-feature modelling, but it is a known out-of-time AAER study.

## 4. Ensembling, stacking and calibration with out-of-time gains

### Takeaway
Evidence for ensembles is mostly from Chinese data with unclear or sliding-window splits. Imbalance-aware boosting (RUSBoost, CUSBoost) beats plain AdaBoost and XGBoost on average, and a fused (averaged) ensemble of the boosters is best. The student's rank-averaged, heterogeneous ensemble already follows this pattern. No paper was found that shows a calibration method changing out-of-time AUC: isotonic calibration is monotone, so it cannot change single-model AUC.

### Cited Findings
- **Rahman & Zhu (Accounting & Finance 63(3), 2023).**
  - Data: 33,544 Chinese A-share firm-years, 1998-2017. Inputs: 12 ratios plus 28 raw items. Models: LR, AdaBoost, XGBoost, CUSBoost and RUSBoost.
  - CUSBoost and RUSBoost beat AdaBoost and XGBoost on average. An "algebraic fused model" combining the four boosters had the highest mean AUC and AUPR.
  - Uses a train/test split, with a sliding window in a supplementary test. The exact test years and serial-fraud handling are not stated in the abstract. — [IDEAS/RePEc](https://ideas.repec.org/a/bla/acctfi/v63y2023i3p3455-3486.html)
- **A paper using the Bao dataset with 2003-2008 as the out-of-sample test** reports:
  - XGBoost AUC 0.689 vs "f-XGBoost" 0.693 in one setup.
  - 0.700 vs 0.777 in another.
  - The mapping of setups is unclear. — [Semantic Scholar PDF](https://pdfs.semanticscholar.org/cdb6/f8149b80daaf56483cdb48123f0261e3ecf7.pdf)
- Bao et al.'s RUSBoost out-of-sample AUC is commonly cited as about 0.725 for 2003-2008. Another source cites 0.878, possibly from a different setup, and this conflict is unresolved. — [Search-result summaries; RePEc Bao entry](https://ideas.repec.org/a/bla/acctfi/v63y2023i3p3455-3486.html) (weak: aggregator-level)
- The Bao et al. replication page says the variable `p_aaer` handles serial fraud and that tuning used train 1991-1999 and validation 2001 (so a 1-year gap at tuning). The page shows no errata. — [jarfraud.github.io](https://jarfraud.github.io/FraudDetection)

### Inferences
- For the student's design, the remaining ensembling levers with some support are:
  - (a) adding an **imbalance-aware boosting member of a different family**, such as RUSBoost or CUSBoost with shallow trees. These are diverse from LightGBM and XGBoost bags, and the project has RUSBoost already as a baseline, so it is cheap to rank-average.
  - (b) **weight optimisation of the rank-average on a temporal validation fold**, e.g. constrained non-negative weights fit on the latest pre-gap year, instead of equal weights.
  - Both are laptop-feasible, but neither has strict out-of-time fraud evidence of a gain.
- Isotonic or other calibration should be judged on Brier or log-loss, not AUC. Only stacking with a learned meta-model can change AUC, and with about 1% positives a learned stacker risks overfitting. Simple rank-averaging is the safer default.

### Gaps
- No 2019-2026 paper found reports a stacking or meta-learner gain over rank-averaging under a strict gap-based temporal split with serial-fraud masking on US AAER data.
- The full Rahman & Zhu text was paywalled (HTTP 403), so the exact splits and AUC values are unknown.

## 5. Size of reported gains and rigour of evaluation (strict out-of-time vs random splits)

### Takeaway
Reported gains are small and fragile under realistic evaluation. Bao-style out-of-time AUCs sit around 0.69-0.78. Realistic label-delay simulation lowers performance. Bao et al. itself was challenged over AAER/serial-fraud label handling. Large gains, such as AUC above 0.9, come from Chinese datasets or stratified or unspecified splits and should not be compared with the student's 0.748.

### Cited Findings
- **Strict OOT, US.**
  - Zavitsanos et al. used a sliding 3-year window, 1-year-ahead test, and tested 1-2-year gaps.
    - Realistic delay labels make all models perform "similarly poorly".
    - Gaps gave no improvement.
    - It rejects random splits as leaky. — [arXiv 2305.17457](https://arxiv.org/html/2305.17457v1)
  - Bertomeu et al. (RAST 2021) report one- and two-year-ahead prediction, and GBRT detections precede AAER filings by at least one year. — [Springer](https://link.springer.com/article/10.1007/s11142-020-09563-8)
- **Bao-dataset OOT (2003-2008 test).** XGBoost variants are reported at AUC 0.689-0.777, depending on the setup. — [Semantic Scholar PDF](https://pdfs.semanticscholar.org/cdb6/f8149b80daaf56483cdb48123f0261e3ecf7.pdf)
- **Serial-fraud and label construction risk.** Walker (2021) shows that AAER identifier changes between training and test can flip whether a fraud is "detected" out of sample, and devotes a section to serial fraud. — [Walker 2021](https://econjwatch.org/File+download/1185/WalkerMar2021.pdf)
- **Random or stratified, or not stated:**
  - The bankruptcy foundation-model study uses a stratified test subset, with no temporal split stated. — [arXiv 2511.16375](https://arxiv.org/html/2511.16375v1)
  - A China A-share XGBoost study (2007-2023) reports AUC 0.9386, which is not comparable. — [ACM DL](https://dl.acm.org/doi/10.1145/3789297.3789329)
  - Rahman & Zhu's split is not specified in the abstract. — [IDEAS](https://ideas.repec.org/a/bla/acctfi/v63y2023i3p3455-3486.html)
  - The hidden-fraud graph paper gives no split detail in the abstract. — [arXiv 2502.19305](https://arxiv.org/abs/2502.19305)
- Text models can overfit to company identity: company name alone gives almost the same performance as full text. This is a leakage symptom relevant to serial fraud. — [arXiv 2305.17457](https://arxiv.org/html/2305.17457v1)

### Inferences
- The student's +0.043 mean yearly AUC over RUSBoost (0.748 vs 0.705) under an expanding window, 2-year gap and serial masking is already at or above what the strict-OOT literature reports. Incremental method gains of +0.002-0.01 are the realistic target.
- Overall feasibility ranking for remaining untested ideas (laptop, free tools), from highest expected value per effort:
  1. RUSBoost or CUSBoost member added to the rank-average, with temporally fitted ensemble weights. This is cheap, supported by ensemble evidence, but not strict-OOT US evidence.
  2. Delay-aware down-weighting of recent-year negatives using an empirical AAER-lag distribution. This is cheap and conceptually new relative to the tested list, but has no published OOT gain.
  3. Bagged TabPFN-2.5 or TabICL on balanced contexts as an extra ensemble member. This needs a free Colab/Kaggle GPU. The financial evidence says it loses standalone; it might add diversity.
  4. Audit-variable inputs (Bertomeu). This has the best published support, but needs Audit Analytics data, which may not be free.
  5. Confident-learning removal of suspect negatives. This is cheap, but the prior is low given the iFPU failure.
  6. Multi-task with restatements. This needs AA data, with no published evidence and noisy labels.

### Gaps
- Exact Bertomeu et al. AUC values, sample years and whether a gap was used could not be read (paywalled).
- I could not confirm whether Bao et al. issued a formal correction after Walker (2021), or what corrected figures they reported.
- No meta-analysis or benchmark compares methods on a common US AAER protocol with gap and serial-fraud masking. Cross-paper gain comparisons are therefore unreliable.

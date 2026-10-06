DepoRank

Term Deposit Campaign Prioritization

Final Data Mining Project Report

COMP6140001 Data Mining
BINUS University
Academic Year 2026/2027

Group member full names and NIM have not been supplied. Add verified details in configs/submission.json and regenerate this report before submission.

Prepared October 6, 2026

Evidence: frozen run 20261005T141242Z_4ac2cd71

Scope: presentation slides, dataset, source code, and this PDF report. Proposal, demo video, and presentation delivery are managed separately by the project team.

Report overview

# Executive summary

DepoRank addresses a campaign planning problem: when contact capacity is limited, which historical contact observations should be prioritized for a term deposit offer? The application provides a locally runnable ranking prototype and an auditable comparison of five data mining models.

Gaussian Naive Bayes was chosen before opening test results, with selection Average Precision (AP) 0.2318. On the final 8,238-row test block, its AP is 0.5076, ROC-AUC 0.6667, and Precision@824 62.14%. The top 824 priorities contain 512 historical yes outcomes, a lift of 2.02 over the test block's base rate.

These results support retrospective ranking utility, not reliable future probabilities or a guaranteed commercial benefit. The yes prevalence rises from 4.81% in train to 30.83% in test. GaussianNB's threshold accuracy is only 46.77%, and its test Brier score is 0.2546. Ranking performance and calibration must therefore be assessed separately.

| Chapter | Content |
| --- | --- |
| 1 Introduction | Case, stakeholder need, objectives, scope, and KDD learning outcomes |
| 2 Theoretical Basis | Data concepts, five model families, calibration, and metric rationale |
| 3 Methods | Data audit, pre-contact features, temporal splits, tuning, and frozen selection |
| 4 Implementation and Evaluation | Architecture, model results, capacity policy, drift, interpretation, and software quality |
| 5 Conclusion and Suggestions | Findings, limitations, and a responsible next experiment |
| 6 Appendix | Public link targets, reproducibility, checksums, references, and publication status |

Submission caveat: supply real cover identities and verify the public slide/data/source links before uploading. A public repository does not publish unpushed local files.

Chapter 1 Introduction

# Case study and objectives

## Business problem

A bank marketing team operates with a finite calling budget. Calling every eligible observation is not possible, while contact histories and customer attributes may contain useful patterns. DepoRank estimates the association between pre-contact attributes and historical term deposit subscription, then sorts observations for a capacity-constrained campaign list.

The stakeholder is a campaign analyst or supervisor. The core decision is how to allocate a fixed number of contacts, not whether a person qualifies for credit. A score is an estimated historical probability, a threshold label is a classification output, and a contact priority is a separate operational policy.

## Project objectives

The project implements the full KDD process, compares at least five distinct model families under the same feature/split protocol, justifies imbalance-aware and capacity-based metrics, and exposes validated single/batch scoring through a reproducible local app. Dataset provenance, configuration, model selection, and final-test evidence must remain traceable.

## Scope and success criteria

Success means that data and code can be inspected, the model comparison is leakage-aware, the final decision is fixed before test, capacity selects exactly K records, and the app produces understandable outputs with stable failure handling. Predictive scores need not be high for the experiment to be methodologically valid; weak results and drift must be reported rather than hidden.

| Learning outcome | Project evidence |
| --- | --- |
| LO 1 Data and preprocessing concepts | Dataset dictionary, audit, warehouse distinction, KDD stages, and leakage exclusions |
| LO 2 Apply data mining techniques | Five trained model families, temporal CV, calibration, metric comparison, and running application |
| LO 3 Trends and research frontiers | Temporal drift, frozen calibration, subgroup checks, responsible evaluation, and prospective monitoring recommendations |

Chapter 2 Theoretical Basis

# Data concepts and the KDD process

A dataset is an organized set of observations and attributes. Here, categorical fields describe customer/contact history and numeric fields encode age, earlier contact counts, and time since a previous campaign. The binary target records whether a term deposit subscription occurred. Data quality includes schema validity, categorical domains, consistency, missingness, duplicates, and provenance.

A data warehouse integrates historical data from operational sources for analysis, commonly with curated dimensions and facts. This prototype uses one public CSV and a local artifact registry; neither its raw folder nor its JSON manifests constitute an enterprise data warehouse. A warehouse is not necessary for the one-source academic experiment, although a production design could integrate permissioned customer snapshots, contact events, and outcomes.

| KDD stage | Application in DepoRank |
| --- | --- |
| Data selection | Choose the official full dataset and pre-contact core attributes; pin checksum and original order. |
| Preprocessing | Parse and validate records; audit duplicate/unknown/inconsistent values; reject malformed input. |
| Transformation | Separate the pdays sentinel from elapsed days; encode categories and scale numeric values inside training folds. |
| Data mining | Search five classifier families using expanding-window CV, then calibrate frozen base models. |
| Interpretation and evaluation | Select/lock before test; compare discrimination, calibration, and capacity utility; inspect importance and subgroups. |

Leakage occurs when a model or preprocessing step receives information unavailable at the decision point or derived from future evaluation data. Fitting an encoder before CV, using post-call duration, tuning on final test, or choosing a new winner after viewing test would contaminate the evidence. The project's separate temporal blocks and pipeline fitting rules address these risks but do not eliminate missing as-of timestamps or unobserved customer overlap.

## Research context

Modern trustworthy evaluation includes out-of-time validation, probability calibration, data/version lineage, drift monitoring, and subgroup evidence. DepoRank demonstrates these concerns at prototype scale rather than claiming state-of-the-art banking performance. The related telemarketing study provides domain context, not a directly comparable accuracy benchmark for this stricter feature/split protocol [1, 2].

Chapter 2 Theoretical Basis

# Five model families and calibration

| Model | Mechanism and limitations |
| --- | --- |
| Logistic Regression | Models log-odds as a linear function of encoded features. Regularization controls coefficient size; it offers a compact baseline but misses nonlinear interactions unless features encode them. |
| Gaussian Naive Bayes | Uses class-conditional Gaussian likelihoods and conditional independence. Fast and parsimonious, but correlated/one-hot features violate its simplifying assumptions; calibration is still required. |
| Decision Tree | Recursively partitions feature space into rules. Captures nonlinear effects and interactions, but deep trees overfit and can produce unstable probability estimates. |
| Random Forest | Averages many randomized decision trees. Reduces variance and captures interactions, but costs memory and does not guarantee robustness to temporal distribution shifts. |
| Gradient Boosting | Builds additive trees sequentially to reduce a loss. Captures nonlinear relationships; learning rate, depth, and estimator count control complexity and fit cost. |

All five families receive the same core raw features and chronological blocks. Their preprocessing stays inside the pipeline. A dummy prior model predicts the training class prevalence, while seeded random ranking tests a no-information ordering. Neither baseline counts as a sixth learned model family.

## Calibration

A model can rank observations well while estimating probabilities poorly. Sigmoid calibration learns a mapping from base scores to outcome likelihood on the independent calibration block. FrozenEstimator prevents recalibration from refitting the already trained base model. Calibration quality is then inspected using Brier loss and reliability bins, not assumed from the use of a calibrator [3].

Training class weighting is part of selected model configurations, not a replacement for independent calibration or minority-sensitive evaluation. No synthetic oversampling is added. Keeping chronology intact is more important than producing a superficially balanced random split.

## Temporal validity

The official full CSV is date-ordered, but individual exact decision timestamps and unique customer identifiers are not supplied. Preserving row order approximates an out-of-time experiment; it cannot prove strict customer-group separation or verify every field's as-of availability. Production validation would need both.

Chapter 2 Theoretical Basis

# Evaluation metrics and stakeholder relevance

| Metric | Reason for inclusion |
| --- | --- |
| Average Precision | Primary search/selection metric. Summarizes precision at recall increments, emphasizing positive-class retrieval without relying on a fixed threshold. |
| ROC-AUC | Threshold-independent discrimination. Useful context, but may obscure weak precision when positive outcomes are uncommon. |
| Precision and Recall at K | Directly answer whether limited campaign slots concentrate yes outcomes and how many positives remain outside capacity. |
| Lift at K | Normalizes Precision@K by the evaluation block's base rate. Values above one indicate concentration above that block's average, not causal treatment uplift. |
| Brier and reliability bins | Evaluate probability error and calibration. Low probability loss is distinct from high ranking utility. |
| F1 and confusion matrix | Describe the classification decision at the selection-frozen threshold, including false positives/negatives. |
| Accuracy | A supplementary threshold metric, always interpreted against class prevalence and the majority baseline. |

Definitions: Precision@K = TP within K / K; Recall@K = TP within K / all positives; Lift@K = Precision@K / prevalence; Brier = mean((p_yes - y)^2). F1 is the harmonic mean of threshold precision and recall. AP follows average_precision_score, not trapezoidal integration [4].

At fraction q, K = ceil(q*N). Ranking uses descending p_yes and ascending input row ID for ties. A decision threshold is not a contact quota: changing q must not retune probabilities or classification thresholds. Undefined metrics are null with explicit reasons rather than fabricated values.

The choice of AP is tied to finding rare historical positives; Precision@K and Lift@K align with the campaign's fixed-capacity decision. This choice does not establish monetary ROI because contact cost, deposit margin, and causal incremental response are absent.

Chapter 3 Methods

# Dataset selection and audit

The selected source is UCI Bank Marketing, bank-additional-full, distributed with its official English dictionary. It contains 41,188 contact observations, 20 original inputs, and target y, from a Portuguese bank between May 2008 and November 2010. The full file is used rather than the randomly selected 4,119-row sample. Its date-ordered original positions support the project's chronological proxy split [1].

| Audit item | Recorded result | Treatment |
| --- | --- | --- |
| Rows and columns | 41,188 and 21 | Enforce the expected variant and preserve original order. |
| Target | 4,640 yes; 36,548 no | Map yes=1, no=0; no target enters features. |
| Exact duplicate rows | 12 | Record in audit; do not silently deduplicate. |
| Errors and warnings | 0 errors; 4,110 warnings | Warnings flag inconsistencies, not automatic corrections. |
| Unknown default | 8,597 | Retain the explicit valid category. |
| Unknown education | 1,731 | Retain the explicit valid category. |
| Unknown housing and loan | 990 each | Retain the explicit valid category. |
| Unknown job and marital | 330 and 80 | Retain the explicit valid category. |

The official CSV is kept unchanged. The dataset SHA-256 is checked before training, and its dictionary hash and source URL are included in the ingestion manifest. Duplicate rows can reflect recurring observations rather than safe-to-delete customer duplicates. No unique customer identifier is available, so customer-level grouping is a stated limitation.

At app input time, parsing and domain validation occur before scoring. Invalid CSV batches are rejected entirely with a downloadable row/field error report. Invalid values are never silently changed to unknown. Valid but inconsistent values receive warnings that remain visible to the operator.

Dataset and dictionary checksums are printed in Chapter 6; the complete raw-field dictionary accompanies the dataset submission ZIP.

Chapter 3 Methods

# Feature availability and transformation

| Core input | Meaning and handling |
| --- | --- |
| age | Integer age; application range 1-120. |
| job, marital, education | Categorical profile attributes with explicit official categories and supported unknown values. |
| default, housing, loan | Credit-default and loan indicators; categorical no/yes/unknown, not a credit approval decision. |
| pdays | Previous-campaign elapsed days; 999 is represented by an availability flag and zero pdays_value. |
| previous | Nonnegative previous-campaign contact count. |
| poutcome | Previous campaign outcome: failure, nonexistent, or success. |

The ten raw features become eleven intermediate fields because pdays is replaced by previously_contacted and pdays_value. When pdays=999, both values become zero; for pdays=0, previously_contacted remains one. One-hot encoding further expands categorical fields. The numeric scaler and encoder are fitted inside each training fold and are then reused without fitting on calibration, selection, or test.

## Excluded fields and leakage rationale

duration is excluded because the final call duration is unknown before the contact. campaign includes contacts in the current campaign, including the last contact, and is conservatively excluded from the pre-contact baseline. y and row_id are never features. Current-contact contact/month/day_of_week and all economic context fields are excluded from core_v1; exact point-in-time availability is not documented for a production use case.

The optional context_v1 definition remains an unrun retrospective extension. No comparison or performance claim is made for it. Expanding the completed baseline after observing test results would require a new version and independent holdout, not a cosmetic change to the existing feature manifest.

## Validity boundary

Removing obvious leakage does not prove all remaining inputs are historically point-in-time correct. A real deployment would need consented customer snapshots at the proposed calling time, event timestamps, and customer IDs. The report distinguishes this limitation from schema correctness.

Chapter 3 Methods

# Chronological validation and model search

| Block | Original positions | Rows | Yes | Prevalence |
| --- | --- | --- | --- | --- |
| Train | [0, 24712) | 24,712 | 1,188 | 4.81% |
| Calibration | [24712, 28831) | 4,119 | 418 | 10.15% |
| Selection | [28831, 32950) | 4,119 | 494 | 11.99% |
| Test | [32950, 41188) | 8,238 | 2,540 | 30.83% |

No shuffling occurs. Original CSV positions are a temporal proxy rather than exact dates or unique customer identities. The fixed 60/10/10/20-style boundaries ensure full, disjoint row coverage. Duplicates and cross-block overlap are recorded in split_manifest.json; the limited duplicate overlap is not silently removed.

| Inner fold | Fit positions | Validation positions |
| --- | --- | --- |
| 1 | [0, 6178) | [6178, 12356) |
| 2 | [0, 12356) | [12356, 18534) |
| 3 | [0, 18534) | [18534, 24712) |

Each fold uses past training positions and a later validation segment, with gap=0 and both classes required. Grid search uses AP and raises errors instead of silently accepting failed fits. After search, each selected base model is fitted on the full train block, then left frozen for calibration.

| Family | Grid candidates | Parameters searched |
| --- | --- | --- |
| Logistic Regression | 6 | C {0.1,1,10}; class_weight {None,balanced} |
| GaussianNB | 3 | var_smoothing {1e-11,1e-9,1e-7} |
| Decision Tree | 16 | depth {3,6,10,None}; leaf {1,20}; class_weight |
| Random Forest | 8 | depth {8,None}; leaf {1,10}; class_weight; 200 trees fixed |
| Gradient Boosting | 8 | trees {100,200}; rate {0.05,0.1}; depth {2,3} |

Total: 41 candidates x 3 folds = 123 fold fits, excluding final base refits and calibration. Seed=42 and detailed fixed settings are in configs/experiment.yaml. The protocol includes no repeated independent split experiments or confidence intervals.

Chapter 3 Methods

# Calibration and frozen selection

Sigmoid calibration is fitted only on positions [24712,28831), using FrozenEstimator to preserve trained base estimators. Selection positions [28831,32950) then supply probability/ranking metrics, model comparison, and threshold selection. The final test block remains unused until the selection lock is written.

| Model | Selection AP | Precision at 412 | Brier | Threshold |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.1773 | 0.2184 | 0.1068 | 0.10 |
| Gaussian Naive Bayes | 0.2318 | 0.2961 | 0.1050 | 0.09 |
| Decision Tree | 0.1605 | 0.1893 | 0.1054 | 0.10 |
| Random Forest | 0.1307 | 0.1214 | 0.1058 | 0.09 |
| Gradient Boosting | 0.1768 | 0.2136 | 0.1048 | 0.06 |

GaussianNB is the selection winner with AP 0.2318. The next strongest selection AP is Logistic Regression at 0.1773, well outside the practical tie interval of 0.005. Tie-breakers therefore do not alter this winner. They are nevertheless specified in advance: higher Precision@10%, lower Brier, lower median inference latency, and model ID.

Each model's decision threshold maximizes F1 on the selection block over 101 grid points from zero to one; a threshold tie prefers the higher threshold. GaussianNB's frozen threshold is 0.09. This relatively low classification threshold does not enforce a 10% contact quota; capacity is applied independently by ranking.

## Freeze and final evaluation

selection_lock.json pins the dataset/config/split hashes, selection metrics hash, all candidate model hashes, winner, feature set, thresholds, and policy version. The active run pointer references this locked winner. Final test evaluates all fixed candidates and baselines once, without choosing a new winner or refitting any model.

Changing the best-looking model after opening test results would turn test into another selection set. Even where another candidate has higher test Precision@K, the app continues to use GaussianNB. Saved-prediction reproduction checks equivalence; it does not authorize a second final-test run.

Chapter 4 Implementation and Evaluation

# Application architecture and workflow

| Layer | Existing implementation |
| --- | --- |
| Entry point | app.py configures Streamlit; src/bank_marketing/ui/pages.py renders seven English pages. |
| Data and features | data/ingest.py and audit.py parse/validate; splits.py preserves chronology; feature transformers fit inside pipelines. |
| Machine learning | ml/train.py, calibrate.py, selection.py, final.py, and registry.py manage training, freeze, evaluation, and verified loading. |
| Services | services/scoring.py validates and predicts; ranking.py selects exact K; export.py emits sanitized CSV/JSON with provenance. |
| Storage and evidence | Atomic writes, manifests, checksums, config_resolved.yaml, saved predictions, environment.json, and selection_lock.json. |
| Submission tooling | A checksum-pinned compressed demo, safe restore script, English report/deck builders, and four-file packaging script. |

The architecture separates an offline experiment runner from a serving application. The app consumes verified, frozen pipelines and evidence instead of running model search during interaction. The raw dataset, model parameters, winner, probabilities, thresholds, and integrity-protected files retain their recorded provenance.

## Input to output

A single-record form or UTF-8 CSV supplies ten core inputs. CSV upload supports comma/semicolon delimiters within 20 MB and 50,000 rows. Validation is all-or-nothing. Successful scoring produces p_yes and a fixed-threshold label; capacity K or fraction q then chooses exactly K highest scores. Ties use input row order. CSV/JSON downloads include model/run provenance, and CLI exports include a metadata sidecar.

Changing K/q only reranks cached scores. Changing data, mode, model identity, or threshold invalidates stale results. Invalid uploads clear old results. No training occurs on Streamlit reruns. Documentation remains accessible when the registry is empty or corrupt, while prediction is disabled.

Session cleanup removes uploaded inputs/results from session state. New CSV uploads never become model artifacts; unknown joblib/pickle uploads are not accepted. Trusted model loading is fail-closed on hash or environment mismatch. A cryptographic checksum detects modification but does not make an untrusted pickle safe.

Chapter 4 Implementation and Evaluation

# Model comparison on the frozen final test

All rows below use the same 8,238-observation final test block, the same core_v1 definition, and selection-frozen thresholds. Capacity q=0.10 produces K=824. These numbers come directly from the verified run's test_metrics.csv; rounding here is for readability only.

| Model | AP | ROC-AUC | P at K | R at K | Lift | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.4130 | 0.5364 | 0.6517 | 0.2114 | 2.1137 | 0.2455 |
| Gaussian Naive Bayes | 0.5076 | 0.6667 | 0.6214 | 0.2016 | 2.0153 | 0.2546 |
| Decision Tree | 0.3893 | 0.5531 | 0.4879 | 0.1583 | 1.5823 | 0.2523 |
| Random Forest | 0.3415 | 0.5769 | 0.2852 | 0.0925 | 0.9250 | 0.2558 |
| Gradient Boosting | 0.4012 | 0.5363 | 0.5510 | 0.1787 | 1.7870 | 0.2419 |

Figure 1 Average Precision across selection and final test blocks

![Figure 1 Average Precision across selection and final test blocks](figures/model_ap.png)

GaussianNB has the highest final-test AP (0.5076) and ROC-AUC (0.6667), consistent with its pre-test selection. Logistic Regression has better top-K precision (0.6517), while Gradient Boosting has the lowest learned-model Brier (0.2419). No model dominates all objectives; test findings do not trigger reselection.

Random Forest's test Lift@K is 0.9250, below the test base rate despite being a flexible ensemble. This is a valid observed result, not removed from the comparison. More complex models are not assumed to be more robust to temporal drift.

Diagnostic baselines: dummy prior has test AP 0.3083 and Brier 0.2810; seeded random ranking has AP 0.3094 and Lift@K 0.9879. Dummy probabilities tie, so deterministic row-order top-K metrics are not a randomized expected baseline. Random ranking has no calibrated-probability or classification metrics.

Chapter 4 Implementation and Evaluation

# Capacity results and classification tradeoffs

| GaussianNB at q 10 percent | Observed test outcome |
| --- | --- |
| Evaluation population | 8,238 contact observations; 2,540 historical yes outcomes |
| Capacity | ceil(0.10 x 8,238) = 824 priorities |
| Yes within priorities | 512 / 824 = 62.14% Precision@K |
| No within priorities | 312 / 824 = 37.86% of priorities |
| Yes outside priorities | 2,028; Recall@K = 512 / 2,540 = 20.16% |
| Lift over block prevalence | 62.14% / 30.83% = 2.0153 |

A finite quota necessarily leaves many positives outside the calling list. At q=10%, the model retrieves roughly one fifth of historical positives, not all of them. Increasing q changes this operational tradeoff but does not establish how many extra subscriptions calling would cause. An analyst must choose capacity from a real budget and evaluate a new prospective campaign.

## The frozen threshold is a different decision

| Actual class | Predicted no | Predicted yes |
| --- | --- | --- |
| no | 1,793 | 3,905 |
| yes | 480 | 2,060 |

At threshold 0.09, precision is 34.53%, recall 81.10%, F1 0.4844, and accuracy 46.77%. The predicted-yes count is 5,965, far above the 824 contact quota. This illustrates why predicted_label and selected_for_contact must remain separate fields.

Always predicting no would have 69.17% test accuracy but recover no positives. This does not excuse the model's poor threshold accuracy; it explains why the report includes both ranking metrics and threshold errors instead of using one favorable number.

There is no observed deposit value, contact cost, or randomized no-contact control group. The 512 historical yes outcomes are not a forecast, profit estimate, or causal incremental response.

Chapter 4 Implementation and Evaluation

# Drift and probability calibration

Figure 2 Observed subscription prevalence changes across chronological blocks

![Figure 2 Observed subscription prevalence changes across chronological blocks](figures/prevalence.png)

Target prevalence is 4.81% in train, 10.15% in calibration, 11.99% in selection, and 30.83% in test. This strong shift changes the meaning of a learned probability mapping and the behavior of a fixed threshold. AP values across blocks have different base rates and should not be interpreted as a pure improvement in model quality.

Figure 3 GaussianNB reliability bins on selection and test using the frozen calibrator

![Figure 3 GaussianNB reliability bins on selection and test using the frozen calibrator](figures/calibration.png)

Calibration is a measured property, not guaranteed by applying sigmoid calibration. GaussianNB's test Brier score is 0.2546 versus 0.1050 on selection. The reliability plot shows that calibration does not transfer cleanly to the later block. The bins are descriptive estimates and no confidence intervals are claimed.

A fresh deployment should monitor feature distributions, outcome prevalence, calibration, and capacity metrics as labels arrive. Recalibrating on test now would contaminate this academic baseline. An approved update needs new calibration/selection data and an independent out-of-time holdout.

Chapter 4 Implementation and Evaluation

# Interpretation and subgroup evidence

Figure 4 Five-repeat permutation importance for the selection block only

![Figure 4 Five-repeat permutation importance for the selection block only](figures/importance.png)

The strongest mean AP decreases after permutation are poutcome (0.0396), pdays (0.0327), and job (0.0239). These findings suggest that previous campaign history and employment category contribute to the fitted model's ranking on selection. The five permutation repeats quantify sensitivity within that block, not repeated independent model experiments.

loan has a slightly negative mean importance (-0.0002). Negative values are retained rather than hidden. Correlated attributes and the Naive Bayes independence assumption limit attribution: permutation importance is neither a causal explanation nor proof that modifying a customer's attribute would improve subscription.

## Subgroups and responsible interpretation

The recorded interpretation report checks age groups, job, marital status, and education on selection. It includes sample sizes, global-capacity selection rates, subgroup precision/recall, and calibration bins. Groups with fewer than 100 observations are explicitly marked as insufficient evidence. Small or correlated subgroup summaries do not establish fairness.

Customer profile attributes may reflect sensitive social differences. Before production use, a bank should define lawful/ethical use constraints, audit harmful disparities with sufficiently sized current data, and document mitigation decisions. DepoRank does not make credit or entitlement decisions and does not claim regulatory compliance.

Interpretation is confined to selection and does not change the feature set, winner, threshold, or test baseline. The complete machine-readable evidence is in the demo archive's interpretation.json.

Chapter 4 Implementation and Evaluation

# Software verification and local performance

The recorded software validation contains 244 passing tests, 0 failures, 0 errors, and 0 skipped tests. Runtime was 36.04 seconds. The evidence summary is reports/software_quality.json. Software test success is not a claim about prediction quality or production readiness.

| Verification area | Examples covered |
| --- | --- |
| Data contracts | Malformed/ambiguous CSV, duplicate headers, invalid values, limits, warnings, and raw hashes. |
| Leakage and models | Fold-specific encoders, sentinel handling, expanding-window validation, calibration freeze, thresholds, and model family definitions. |
| Registry and safety | Lock/model hashes, active-pointer corruption, path escape, environment mismatch, and protected output paths. |
| App workflow | Seven pages, single/batch input, exact K, reranking without retraining, stale-result invalidation, session cleanup, and errors. |
| Submission tooling | Trusted demo restoration, checksum failure, safe extraction, non-overwrite policy, and English navigation labels. |

| Recorded local measurement | Value |
| --- | --- |
| Warm registry startup | 0.180 seconds |
| Single scoring median over 20 repeats | 0.020 seconds |
| 10,000-row parse/validate/predict/rank | 0.242 seconds |
| Sampled peak process RSS | 191.6 MiB |

Measurements come from the original October 5, 2026 experiment on Python 3.11.16, Windows, 20 logical CPUs, and approximately 23.7 GiB RAM. Warm startup excludes cold Python/Streamlit startup. RSS polling every 5 ms is not complete allocation instrumentation. These are local observations, not cross-machine benchmarks or banking SLAs.

The source ZIP provides setup instructions, exact dependencies, tests, the unchanged dataset, and a compact trusted archive containing all five models. Restoration avoids retraining for a classroom demo and refuses to overwrite conflicting local artifacts. Final human checks still include the actual presentation computer, keyboard navigation, offline execution after setup, and no-login public downloads.

Chapter 5 Conclusion and Suggestions

# Findings and recommendations

## Conclusion

The completed project implements all five KDD stages and compares five distinct classifier families using a shared, leakage-aware temporal protocol. Gaussian Naive Bayes is the selection-frozen winner. On final test it reaches AP 0.5076 and concentrates 512 historical yes outcomes into 824 priorities, yielding 62.14% Precision@K and 2.02 lift against the later block's 30.83% base rate.

The evidence also contains important weaknesses. Probability calibration degrades under a large prevalence shift; the chosen classification threshold has poor accuracy; Random Forest's top-K lift is below one; and another model can outperform the winner on one test metric. These findings are disclosed rather than resolved by post-test tuning. The project demonstrates an auditable retrospective decision workflow, not a bank-ready or causal marketing system.

## Limitations

The source is historical Portuguese data, not current Indonesian customer data. Contact observations cannot be verified as unique customers. Exact as-of timestamps, consent records, real campaign costs/margins, and randomized treatment outcomes are absent. Only one fixed chronological experiment is reported; uncertainty intervals and external validation are not available. The optional context experiment is unrun.

## Suggested next experiment

First obtain authorized current data and define the real decision time, eligibility rules, customer grouping, costs, and capacity. Construct point-in-time features with customer IDs and new out-of-time holdouts. Retain a simple baseline, compare more models only under a predeclared search budget, and estimate uncertainty with an appropriate temporal procedure.

Next validate calibration and subgroup effects on sufficiently sized data. Monitor prevalence, score distributions, capacity precision, label delay, and calibration. Refresh calibration or retrain only through a versioned protocol with new independent evaluation data. Explain model changes to the operator and preserve lineage.

If the business question is incremental response to being contacted, use a controlled prospective campaign and an appropriate causal/uplift design. Classification on historical subscription labels alone cannot identify that effect. A production rollout requires business, privacy, ethics, and operational approval outside this academic project's scope.

Chapter 6 Appendix

# Public links and reproduction

The following publication targets are intended for anyone to download/view without login. The repository was confirmed public on October 6, 2026. Newly prepared slide/data/source files must still be pushed before their links can satisfy the guideline; local file existence does not prove public availability.

Presentation slides: https://github.com/DlegendZ/DepoRank/raw/refs/heads/main/presentation/slides.pptx

Project dataset: https://github.com/DlegendZ/DepoRank/raw/refs/heads/main/data/raw/bank-additional-full.csv

Source code repository: https://github.com/DlegendZ/DepoRank

Official dataset download fallback: https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip

Official dataset documentation: https://archive.ics.uci.edu/dataset/222/bank+marketing

## Submission bundle

| File | Purpose |
| --- | --- |
| DepoRank_Presentation.pptx | Editable stakeholder presentation |
| DepoRank_Dataset.zip | Raw CSV, official dictionary, provenance, and attribution |
| DepoRank_Source_Code.zip | Runnable repository snapshot, tests, docs, data, and frozen demo |
| DepoRank_Report.pdf | This six-chapter report |

Reproduction on Python 3.11: install requirements.lock, install this project editable with no dependency resolution, run scripts/restore_demo.py, verify run 20261005T141242Z_4ac2cd71, then launch Streamlit with app.py. README.md includes exact Windows and Unix commands. The restore archive keeps locked evidence byte for byte; historical metadata may retain original language while the report/UI provide English explanations.

Before submission, add real member names/NIM, regenerate the artifacts, commit/push to the public main branch, and check all links signed out. scripts/check_public_links.py performs anonymous HTTP checks; a human must also open/download the results. The team separately handles proposal, demo video, and live presentation. No such activity is falsely recorded as completed.

Chapter 6 Appendix

# Evidence identifiers and references

Frozen run ID: 20261005T141242Z_4ac2cd71

Dataset SHA-256: 74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8

Resolved configuration object SHA-256: ff7af328e9c3eb10783fff62395e0c80cd00fe30d52093d1e315751a0af0a756

Selection-lock payload SHA-256: 4ae7592560bc2e8a67fd5a6b1c83bdc8f1d9123338a5b440d1dda0fb0f027850

Configuration and lock identifiers use storage.object_hash: SHA-256 over UTF-8 canonical JSON with sorted keys and compact separators. The lock's own lock_sha256 field is excluded. These are not raw YAML/JSON file checksums; file-level checksums remain in run_manifest.json.

Machine-readable evidence includes selection_metrics.csv, test_metrics.csv, model metadata, saved predictions, evaluation curves, split_manifest.json, environment.json, benchmark.json, and interpretation.json in the restored run. Hash-locked evidence is not edited for translation. Generated charts and report text can be refreshed without changing the locked model decision.

## References

[1] UCI Machine Learning Repository. Bank Marketing dataset and official bank-additional-names dictionary. Creators: Sergio Moro, Paulo Cortez, and Paulo Rita. Dataset distributed under CC BY 4.0.

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing

[2] Moro, S., Cortez, P., and Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, 62, 22-31. DOI 10.1016/j.dss.2014.03.001.

Source: https://doi.org/10.1016/j.dss.2014.03.001

[3] scikit-learn. Probability calibration and FrozenEstimator documentation. Used to define separate calibration of already fitted estimators.

Source: https://scikit-learn.org/stable/modules/calibration.html

[4] scikit-learn. Model evaluation documentation, including Average Precision, ROC-AUC, classification metrics, and Brier loss.

Source: https://scikit-learn.org/stable/modules/model_evaluation.html

[5] scikit-learn. TimeSeriesSplit documentation. Used for expanding-window validation restricted to the train block.

Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

Course source: user-supplied Data Mining final project guideline.docx, COMP6140001, academic year 2026/2027. Requirements applied: full KDD, at least five models, justified metrics, six report chapters, cover identities, and public slides/dataset/source links. The original guideline is preserved locally and excluded from public source packaging.

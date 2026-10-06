# DepoRank system requirements

## Purpose and scope

DepoRank supports retrospective term deposit campaign analysis and capacity-constrained contact ranking. The observation is a contact record, not a uniquely identified customer. The local academic application includes ingestion/audit, leakage-safe feature engineering, comparison of five model families, frozen selection, single/batch scoring, ranking, evidence views, and export.

The system is not a production bank integration, credit decision tool, causal uplift estimator, ROI calculator, consent manager, or enterprise data warehouse. Deployment, public access, real identities, presentation activities, proposal, and video are handled separately by the team. The course guideline controls report structure and deliverables, not the model's API contract.

## Data contracts

The only completed experiment uses the UCI `bank-additional-full` CSV:

- 41,188 rows, 21 columns, semicolon delimiter, UTF-8-compatible text.
- SHA-256: `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8`.
- Original order, values, and duplicate rows are preserved.
- Target `y` is exactly `no` or `yes`. Raw row identity is zero-based original CSV position.
- Audit records 12 full-row duplicates, 0 errors, and 4,110 warnings; warnings are not missing rows or automatic corrections.
- `unknown` is an explicit valid category where the schema allows it. Empty cells, invalid domains, or schema violations are errors.

### Scoring schema

| Column | Domain or meaning |
| --- | --- |
| `age` | Integer 1-120 |
| `job` | `admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown` |
| `marital` | `divorced`, `married`, `single`, `unknown` |
| `education` | `basic.4y`, `basic.6y`, `basic.9y`, `high.school`, `illiterate`, `professional.course`, `university.degree`, `unknown` |
| `default`, `housing`, `loan` | `no`, `yes`, `unknown` |
| `pdays` | Integer 0-999; 999 means no previous campaign contact |
| `previous` | Nonnegative integer, previous campaign contact count |
| `poutcome` | `failure`, `nonexistent`, `success` |
| `client_id` | Optional output identifier, never a feature |

Default scoring accepts ten core features plus optional `client_id`; extra context, target, or leakage columns are rejected. Labeled demo mode accepts the explicitly allowed raw dataset fields and target for demonstration, strips non-core fields with warnings, and never treats upload evaluation as a new final test. All invalid batches are rejected as a whole.

CSV limits are 20,000,000 bytes and 50,000 rows. UTF-8 comma/semicolon parsing rejects duplicate headers, ambiguous dialects, malformed quoting, invalid numeric values, and unexpected fields. Uploaded content never becomes a server filesystem path.

## Feature processing

`core_v1` uses the ten scoring features. `duration` is post-call information and is prohibited. `campaign` includes contacts in the current campaign and is excluded to avoid ambiguous pre-contact availability. `y`, row identity, current-contact timing, and the five economic indicators are not used in the main baseline.

`pdays=999` becomes `previously_contacted=0` and `pdays_value=0`. All other valid pdays values keep `previously_contacted=1`, including zero. This avoids treating 999 as an actual elapsed interval. The transformer creates eleven intermediate features; one-hot encoding further expands categorical features. The numeric scaler and categorical encoder fit inside the training pipeline, never on future blocks or before CV.

`context_v1` is a separate supported retrospective feature definition; it was not trained or used for the submitted baseline. No result is claimed for it.

## Training and evaluation protocol

| Block | Half-open original positions | Rows | Role |
| --- | --- | ---: | --- |
| Train | `[0,24712)` | 24,712 | Inner-CV search and base fit |
| Calibration | `[24712,28831)` | 4,119 | Sigmoid probability calibration |
| Selection | `[28831,32950)` | 4,119 | Winner, thresholds, interpretation |
| Test | `[32950,41188)` | 8,238 | One final locked evaluation |

The training block uses three expanding-window `TimeSeriesSplit` folds with gap=0. Search compares 41 parameter candidates and performs 123 fold fits. Logistic Regression, GaussianNB, Decision Tree, Random Forest, and Gradient Boosting are mandatory distinct families. Parameter grids, fixed seed=42, and model parameters are in `configs/experiment.yaml`.

Base estimators are frozen before sigmoid calibration on the separate calibration block. No refitting after calibration or selection is permitted for this baseline. The recorded winning candidate is GaussianNB, threshold 0.09.

Selection maximizes AP. A practical AP tie is within 0.005 of the maximum; tie-breakers are descending Precision@10%, ascending Brier, ascending median prediction latency, and lexical model ID. Thresholds maximize F1 on selection across 101 grid points, with higher thresholds breaking ties. Model selection and threshold selection are distinct decisions.

The selection lock pins dataset/config/split hashes, candidate model hashes, selection metrics, winner, feature set, capacity policy, and thresholds. Final test runs only after freezing; repeated evaluation or overwriting an attempted test is rejected. The registry uses an explicit active pointer, never directory-name sorting.

### Metric definitions

- **AP:** the non-interpolated weighted sum of precision at recall increments, as `average_precision_score` computes it; not trapezoidal PR area.
- **ROC-AUC:** discrimination over thresholds, provided alongside AP rather than replacing imbalance-aware evaluation.
- **Precision@K:** observed yes outcomes within the selected K divided by K.
- **Recall@K:** selected yes outcomes divided by all yes outcomes in that evaluation block.
- **Lift@K:** Precision@K divided by block target prevalence.
- **Brier:** mean squared probability error; lower is better, but ranking quality does not imply good calibration.
- **Precision/recall/F1/accuracy:** computed using the selection-frozen threshold, not a threshold optimized on test.

Zero-capacity, absent-class, and undefined metrics use null plus explicit reason codes where appropriate, rather than misleading fabricated zero values. Random ranking has no probability calibration or classification threshold. Dummy prior is a baseline, not a sixth learned family; deterministic ties may make its top-K metrics poor.

## Scoring and ranking contracts

Model loading requires verified artifact hashes, consistent lock/model metadata, valid model ID/threshold, and matching Python/model-library versions. Error handling is fail-closed. Unknown joblib uploads and arbitrary model paths are not accepted.

Every score returns stable input row identity, optional client ID, `p_yes`, threshold label, provenance, and warnings. Classification is `p_yes >= threshold`. Ranking uses descending probability and ascending input row identity for ties. K and q are mutually exclusive; K must be an integer in `[0,N]`, q a finite fraction in `[0,1]`, and fractional capacity becomes `ceil(q*N)`. The policy selects exactly K rows independently of predicted labels.

Changing capacity reranks cached scores without model training or prediction. Changing input data, input mode, model/run/hash, or threshold invalidates stale results. Invalid input clears old batch results. Session cleanup clears single/batch results and resets upload/demo widgets.

## UI and export

Seven English pages cover Overview, Data and KDD, Model Comparison, Single Prediction, Batch Priorities, Evaluation and Interpretation, and Documentation. Empty or corrupted registries leave documentation reachable and disable prediction. Curves can render directly from saved evaluation JSON if pre-rendered charts are absent.

CSV/JSON exports include provenance; CLI CSV/JSON output has a metadata sidecar. Top-K-only and original-feature export are explicit options. Numeric probabilities remain numeric in exports. User-provided spreadsheet-like strings are sanitized; legitimate negative numeric values remain numbers. NaN/Infinity cannot be serialized as results. Exports must not overwrite inputs, raw datasets, manifests, or model artifacts.

## Evidence and performance

Unit/integration tests exercise parsing, audits, feature leakage, folds, model families, calibration, selection, thresholds, hashes, environment failures, deterministic ranking, exact capacity, stale-state invalidation, and every app page. The quality summary distinguishes software tests from predictive evaluation.

Recorded local measurements include a warm registry load, 20-repeat median single-observation scoring, a 10,000-row end-to-end batch, and sampled process RSS. These are hardware-specific local observations, not production SLAs or a cold-start measurement. Offline/keyboard usability and live delivery require operator validation.

## Limitations and responsible use

Historical Portugal data from May 2008-November 2010 is not representative evidence for a current Indonesian bank. The strong train/test prevalence shift weakens calibration. Duplicate contact records, missing unique customer IDs, and missing as-of timestamps limit temporal/customer isolation. Protected-attribute associations and small subgroup samples require further review; limited subgroup metrics do not establish fairness.

Production use requires permissioned current data, verified feature availability at decision time, out-of-time and customer-grouped validation, uncertainty assessment, cost/capacity alignment, prospective monitoring, consent controls, subgroup audit, and an approved retraining/calibration protocol with a new holdout. Predictive ranking does not establish an incremental benefit of contacting anyone.

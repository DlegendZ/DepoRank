# DepoRank

**Term deposit campaign analysis and contact prioritization.** DepoRank compares five data mining model families on the UCI Bank Marketing dataset and ranks contact observations within a fixed campaign capacity. The Streamlit app uses a frozen, auditable experiment; opening a page never retrains a model.

This is a local academic prototype for **COMP6140001 Data Mining, BINUS University**. It predicts historical associations, not causal uplift, credit eligibility, or production-ready banking decisions.

## Submission files

| Item | Repository file | Submission copy |
| --- | --- | --- |
| Presentation | `presentation/slides.pptx` | `submission/DepoRank_Presentation.pptx` |
| Dataset | `data/raw/bank-additional-full.csv` and official dictionary | `submission/DepoRank_Dataset.zip` |
| Source code | This repository | `submission/DepoRank_Source_Code.zip` |
| Six-chapter report | `reports/final_report.pdf` | `submission/DepoRank_Report.pdf` |

`presentation/slides.pdf` is an additional viewing copy; the submission folder contains only the four requested deliverables. `reports/final_report.md` is the editable report text. Before uploading, follow [the submission checklist](docs/SUBMISSION_CHECKLIST.md). Member names/NIM and public links must be completed; a public repository alone does not publish unpushed files.

The presentation, demo video, and proposal activities remain the team's responsibility. The build scripts do not generate a proposal or video.

## Run the existing demo

Use **Python 3.11** and the pinned environment in `requirements.lock`. The saved models require matching Python, scikit-learn, NumPy, pandas, and joblib versions; do not bypass the registry's environment checks.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.lock
.venv/Scripts/python.exe -m pip install -e . --no-deps
.venv/Scripts/python.exe scripts/restore_demo.py
.venv/Scripts/python.exe -m bank_marketing.cli verify --run-id 20261005T141242Z_4ac2cd71
.venv/Scripts/python.exe -m streamlit run app.py
```

### macOS or Linux

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python scripts/restore_demo.py
.venv/bin/python -m bank_marketing.cli verify --run-id 20261005T141242Z_4ac2cd71
.venv/bin/python -m streamlit run app.py
```

After dependencies are installed, the app and synthetic demo run locally without network access. The included CSV supplies the public-data example. If an exact wheel is unavailable for another OS, use a matching supported environment or create a new experiment; do not silently load incompatible models.

The archive in `demo/` preserves all five candidate models and their evaluation evidence, while avoiding individual 80 MB random-forest files in Git. The restore script verifies its checksum and refuses to overwrite different existing artifacts. Repeating restore on an unchanged installation is safe.

**Security:** joblib/pickle models can execute code. Restore only the trusted bundle in this repository. Never upload real customer identities, account details, phone numbers, or unknown model files. Uploaded CSV batches remain in session memory; exported files are controlled by the operator.

## App pages and demo workflow

1. **Overview** shows dataset counts, the locked winner, and prevalence drift.
2. **Data and KDD** explains selection, preprocessing, transformation, mining, and interpretation.
3. **Model Comparison** displays inner-CV, selection, and final-test evidence without changing the active model.
4. **Single Prediction** validates a ten-feature observation and returns `p_yes` and a threshold label.
5. **Batch Priorities** accepts a valid CSV, applies K or q capacity, and exports priorities plus provenance.
6. **Evaluation and Interpretation** displays curves, confusion matrices, importance, and subgroup evidence.
7. **Documentation** provides setup, manifests, report/slides downloads, and session cleanup.

For a demo, use `demo_valid.csv` at q=10%, change capacity without rescoring, download results, then select `demo_invalid.csv` to show whole-batch rejection. Labeled public examples require labeled demo mode and are not a new test holdout.

## Inputs and outputs

Unlabeled scoring requires these columns:

```text
age,job,marital,education,default,housing,loan,pdays,previous,poutcome
```

CSV must be UTF-8, delimited by `;` or `,`, and within **20 MB / 50,000 rows**. `client_id` is optional and is an export label, never a model feature. Unknown categories are represented explicitly by `unknown` where permitted. Blank or invalid values are rejected, not silently repaired. `pdays=999` means no previous contact; `pdays=0` means a previous contact today.

Default scoring rejects target/leakage/context columns. Labeled demo mode explicitly removes permitted demo-only columns and reports warnings. Detailed domains and output fields are in [the requirements](docs/REQUIREMENTS.md) and [the data dictionary](data/README.md).

Outputs contain probability, threshold label, deterministic rank, `selected_for_contact`, input row identity, model/run/feature-set identifiers, and validation warnings. Capacity selection is separate from classification: K contacts need not equal the number of predicted yes labels. Ranking ties use input row order. CSV exports sanitize user strings that could trigger spreadsheet formulas.

```powershell
.venv/Scripts/python.exe -m bank_marketing.cli score --input tests/fixtures/demo_valid.csv --output artifacts/example_priorities.csv --q 0.1
```

The CLI writes a metadata sidecar. Errors use stable codes such as `SCHEMA_INVALID`, `ROW_INVALID`, `ARTIFACT_INTEGRITY_ERROR`, and `ENVIRONMENT_MISMATCH`; translated messages do not change these contracts.

## Experiment protocol

- **Dataset:** UCI `bank-additional-full`, 41,188 observations, 20 original inputs and target `y`; 4,640 yes labels (11.27%). Original raw values and row order are preserved. Twelve full-row duplicates are recorded, not silently removed.
- **Features:** `core_v1` uses ten pre-contact attributes. It excludes `duration`, `campaign`, target, row IDs, current-contact timing, and economic context from the main baseline. `context_v1` is not part of the completed experiment.
- **Splits:** train `[0,24712)`, calibration `[24712,28831)`, selection `[28831,32950)`, test `[32950,41188)`. Original CSV order is a temporal proxy, not exact customer/date information.
- **Training:** three expanding-window folds on train only, 41 grid candidates, 123 fold fits. One-hot encoding/scaling is fit inside each training fold.
- **Models:** Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest, and Gradient Boosting. Dummy prior and random ranking are diagnostic baselines, not additional model families.
- **Calibration:** sigmoid calibration on a separate block with `FrozenEstimator`; base models remain frozen.
- **Selection:** highest selection Average Precision (AP); candidates within 0.005 use Precision@10%, Brier score, inference latency, then model ID as tie-breakers. F1 selects each threshold on selection data.
- **Final test:** winner and thresholds are locked before test. No model reselection or refit after viewing test results. Reproduction compares saved predictions without rerunning the final-test stage.

## Observed results

The pre-test winner is **Gaussian Naive Bayes**, with selection AP **0.2318** and fixed threshold **0.09**. On the 8,238-row test block at K=824:

| Model | Test AP | ROC-AUC | Precision@K | Recall@K | Lift@K | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.4130 | 0.5364 | 0.6517 | 0.2114 | 2.1137 | 0.2455 |
| Gaussian Naive Bayes | 0.5076 | 0.6667 | 0.6214 | 0.2016 | 2.0153 | 0.2546 |
| Decision Tree | 0.3893 | 0.5531 | 0.4879 | 0.1583 | 1.5823 | 0.2523 |
| Random Forest | 0.3415 | 0.5769 | 0.2852 | 0.0925 | 0.9250 | 0.2558 |
| Gradient Boosting | 0.4012 | 0.5363 | 0.5510 | 0.1787 | 1.7870 | 0.2419 |

GaussianNB finds **512 historical yes outcomes among 824 priorities**; 312 priorities are no and 2,028 yes outcomes remain outside capacity. These are retrospective outcomes, not guaranteed future subscriptions or revenue. Logistic Regression has higher test Precision@K but does not replace the selection winner. The majority-no test accuracy is 69.17%, above GaussianNB's 46.77%; accuracy alone would obscure ranking utility and poor threshold calibration.

Prevalence increases from **4.81% train** to **30.83% test**. Calibration and production transfer are not established. No bootstrap confidence intervals, repeated independent experiments, actual ROI, consent evidence, or causal treatment effects are claimed.

## Retrain or verify evidence

The frozen demo is the fastest route to run the app. To create a separate experiment, use a new run ID and run these stages in order:

```powershell
.venv/Scripts/python.exe -m bank_marketing.cli ingest
.venv/Scripts/python.exe -m bank_marketing.cli audit
.venv/Scripts/python.exe -m bank_marketing.cli split
.venv/Scripts/python.exe -m bank_marketing.cli train --run-id my_new_run
.venv/Scripts/python.exe -m bank_marketing.cli calibrate --run-id my_new_run
.venv/Scripts/python.exe -m bank_marketing.cli select --run-id my_new_run
.venv/Scripts/python.exe -m bank_marketing.cli freeze --run-id my_new_run
.venv/Scripts/python.exe -m bank_marketing.cli test --run-id my_new_run
.venv/Scripts/python.exe -m bank_marketing.cli activate --run-id my_new_run
```

Training can take minutes and substantial RAM. Do not use the shipped baseline run ID or reevaluate its final test. `configs/experiment.yaml` and `requirements.lock` define the recorded procedure. `bank_marketing.ml.reproduce.verify_prediction_parity` verifies existing saved predictions; it does not choose a new model.

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m bank_marketing.cli benchmark --run-id 20261005T141242Z_4ac2cd71
.venv/Scripts/python.exe -m bank_marketing.cli report --run-id 20261005T141242Z_4ac2cd71
```

Edit real member details in `configs/submission.json` before regenerating the report. `scripts/build_report.py` builds the PDF and editable report text. `scripts/build_slides.mjs` builds editable PowerPoint slides using `@oai/artifact-tool`; this authoring dependency is separate from the Python app. `scripts/package_submission.py` refreshes the four-file submission bundle from versioned source. See each script's help and the checklist for exact commands.

## Repository layout

```text
app.py                     Streamlit entry point
configs/                   Experiment protocol and submission identity
data/                      Public dataset, dictionary, and audit manifests
demo/                      Compressed frozen academic run and checksum
docs/                      English requirements and submission checklist
presentation/              Editable deck and PDF viewing copy
reports/                   Six-chapter PDF, editable text, and evidence summaries
scripts/                   Safe demo restore and artifact builds
src/bank_marketing/         Existing data, features, ML, services, UI, and CLI
tests/                     Unit, integration, UI, and packaging tests
submission/                Generated upload bundle, ignored by Git
```

Local `.venv`, caches, uploads, unpacked model artifacts, build previews, original course reference files, and duplicate source archives are excluded from commits. User-supplied references remain local, unchanged. Frozen run evidence is not rewritten for cosmetic translation; English UI explanations accompany immutable historical metadata.

## Attribution

Data: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The distributed raw CSV is unchanged; derived manifests, splits, models, and charts are project outputs.

Moro, S., Cortez, P., and Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems, 62, 22-31. DOI: [10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001).

Source code uses the existing MIT license in `LICENSE`; the dataset retains its separate attribution and license.

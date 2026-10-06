"""Build the six-chapter English report from verified, frozen experiment evidence."""

from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from bank_marketing.ml.registry import Registry
from bank_marketing.storage import read_json, write_json

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "gaussian_nb": "Gaussian Naive Bayes",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "dummy_prior": "Dummy prior",
    "random_ranking": "Random ranking",
}
NAVY = "#172D43"
TEAL = "#087F83"
GOLD = "#CB9344"


def _figures(run, directory, selection, test, split, interpretation):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    paths = {}

    def save(name, figure):
        figure.tight_layout(pad=1.5)
        paths[name] = directory / f"{name}.png"
        figure.savefig(paths[name], dpi=180, facecolor="white")
        plt.close(figure)

    order = list(MODEL_NAMES)[:5]
    figure, axis = plt.subplots(figsize=(8, 3.8))
    for offset, frame, label, color in ((-0.18, selection, "Selection", TEAL), (0.18, test, "Final test", GOLD)):
        values = [float(frame.loc[model, "average_precision"]) for model in order]
        axis.bar([index + offset for index in range(5)], values, width=0.36, label=label, color=color)
    axis.set(xticks=range(5), xticklabels=["Logistic\nRegression", "Gaussian\nNaive Bayes", "Decision\nTree", "Random\nForest", "Gradient\nBoosting"], ylabel="Average Precision", ylim=(0, 0.62))
    axis.legend(frameon=False)
    save("model_ap", figure)

    figure, axis = plt.subplots(figsize=(8, 3.1))
    labels = ["train", "calibration", "selection", "test"]
    values = [split["counts"][name]["prevalence"] * 100 for name in labels]
    bars = axis.bar(labels, values, color=[NAVY, TEAL, TEAL, GOLD], width=0.55)
    axis.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4)
    axis.set(ylabel="Observed yes prevalence (%)", ylim=(0, 37))
    save("prevalence", figure)

    report = read_json(run / "evaluations/test_gaussian_nb.json")
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    curve = report["curves"]["precision_recall"]
    axes[0].step(curve["recall"], curve["precision"], where="post", color=TEAL)
    axes[0].axhline(report["metrics"]["prevalence"], linestyle="--", color=GOLD, label="Test prevalence")
    axes[0].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1), title="Test precision and recall")
    axes[0].legend(fontsize=8, frameon=False)
    curve = report["curves"]["roc"]
    axes[1].plot(curve["fpr"], curve["tpr"], color=TEAL)
    axes[1].plot([0, 1], [0, 1], "--", color=GOLD)
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1), title="Test ROC")
    save("discrimination", figure)

    figure, axis = plt.subplots(figsize=(8, 3.3))
    for scope, color in (("selection", TEAL), ("test", GOLD)):
        curve = read_json(run / f"evaluations/{scope}_gaussian_nb.json")["curves"]["calibration"]
        axis.plot([item["mean_predicted"] for item in curve], [item["observed"] for item in curve], "o-", label=scope.title(), color=color)
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed yes proportion", xlim=(0, 1), ylim=(0, 1))
    axis.legend(frameon=False, fontsize=8)
    save("calibration", figure)

    figure, axis = plt.subplots(figsize=(8, 3.6))
    importance = sorted(interpretation["permutation_importance"], key=lambda item: item["mean_ap_decrease"])
    axis.barh([item["feature"] for item in importance], [item["mean_ap_decrease"] for item in importance], color=TEAL)
    axis.set(xlabel="Mean AP decrease after permutation on selection")
    save("importance", figure)
    return paths


def build_report(run_id, root="artifacts", project_root=None):
    project = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    registry = Registry(root)
    manifest = registry.verify_run(run_id)
    run = registry.run_path(run_id)
    if manifest["status"] != "tested":
        raise ValueError("The final report requires a completed, frozen test evaluation.")
    identity = read_json(project / "configs/submission.json")
    if identity["run_id"] != run_id:
        raise ValueError("Submission configuration and requested run ID differ.")
    lock = read_json(run / "selection_lock.json")
    split = read_json(run / "split_manifest.json")
    audit = read_json(run / "audit.json")["counts"]
    selection = pd.read_csv(run / "selection_metrics.csv").set_index("model_id")
    test = pd.read_csv(run / "test_metrics.csv").set_index("model_id")
    interpretation = read_json(run / "interpretation.json")
    benchmark = read_json(run / "benchmark.json")
    quality_path = project / "reports/software_quality.json"
    quality = read_json(quality_path) if quality_path.exists() else {}
    output = project / "reports"
    output.mkdir(exist_ok=True)
    publication_file = output / "publication_checks.json"
    publication_verified = publication_file.exists() and read_json(publication_file).get("all_accessible", False)
    figures = _figures(run, output / "figures", selection, test, split, interpretation)
    styles = getSampleStyleSheet()
    styles["Title"].fontName = "Helvetica-Bold"
    styles["Title"].textColor = colors.HexColor(NAVY)
    styles["Title"].fontSize = 32
    styles["Title"].leading = 38
    for name in ("Heading1", "Heading2"):
        styles[name].textColor = colors.HexColor(TEAL)
        styles[name].spaceAfter = 12
    styles["BodyText"].fontSize = 10.2
    styles["BodyText"].leading = 14.6
    styles["BodyText"].spaceAfter = 9
    styles.add(ParagraphStyle("TableBody", fontName="Helvetica", fontSize=8.3, leading=11, textColor=colors.HexColor(NAVY)))
    styles.add(ParagraphStyle("TableHeader", parent=styles["TableBody"], fontName="Helvetica-Bold", textColor=colors.white))
    styles.add(ParagraphStyle("Caption", fontName="Helvetica", fontSize=8.3, leading=11, textColor=colors.HexColor("#536779"), spaceAfter=10))
    story = []
    markdown = []

    def text(value, style="BodyText"):
        story.append(Paragraph(escape(str(value)).replace("\n", "<br/>"), styles[style]))
        markdown.append(str(value) + "\n")

    def heading(title, level=2):
        text(title, "Heading1" if level == 1 else "Heading2")
        markdown[-1] = "#" * level + " " + title + "\n"

    def page(chapter, title):
        if story:
            story.append(PageBreak())
        if chapter:
            text(chapter, "Caption")
        heading(title, 1)

    def table(headers, rows, widths=None):
        data = [[Paragraph(escape(str(value)), styles["TableHeader"]) for value in headers]]
        data += [[Paragraph(escape(str(value)), styles["TableBody"]) for value in row] for row in rows]
        width = A4[0] - 96
        result = Table(data, colWidths=widths or [width / len(headers)] * len(headers), repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F5F6")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(TEAL)),
        ]))
        story.extend([result, Spacer(1, 12)])
        markdown.append("| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(str(value) for value in row) + " |" for row in rows) + "\n")

    def image(name, caption, height):
        story.append(Image(str(figures[name]), width=A4[0] - 96, height=height, kind="proportional"))
        text(caption, "Caption")
        markdown.append(f"![{caption}](figures/{name}.png)\n")

    def link(label, url):
        story.append(Paragraph(f'<b>{escape(label)}</b><br/><link href="{escape(url)}" color="{TEAL}">{escape(url)}</link>', styles["BodyText"]))
        markdown.append(f"{label}: {url}\n")

    winner = test.loc["gaussian_nb"]
    selected = selection.loc["gaussian_nb"]
    model_order = list(MODEL_NAMES)[:5]
    identity_complete = bool(identity.get("members")) and all(member.get("full_name") and member.get("nim") for member in identity["members"])
    story.append(Spacer(1, 90))
    text("DepoRank", "Title")
    text("Term Deposit Campaign Prioritization", "Heading1")
    text("Final Data Mining Project Report", "Heading2")
    text(f"{identity['course']}\n{identity['institution']}\nAcademic Year {identity['academic_year']}")
    if identity.get("class"):
        text("Class " + identity["class"])
    story.append(Spacer(1, 25))
    if identity_complete:
        table(["Group member full name", "Student ID or NIM"], [[member["full_name"], member["nim"]] for member in identity["members"]], [330, 169.28])
    else:
        text("Group member full names and NIM have not been supplied. Add verified details in configs/submission.json and regenerate this report before submission.")
    text("Prepared October 6, 2026", "Caption")
    text("Evidence: frozen run " + run_id, "Caption")
    text("Scope: presentation slides, dataset, source code, and this PDF report. Proposal, demo video, and presentation delivery are managed separately by the project team.", "Caption")

    page("Report overview", "Executive summary")
    text("DepoRank addresses a campaign planning problem: when contact capacity is limited, which historical contact observations should be prioritized for a term deposit offer? The application provides a locally runnable ranking prototype and an auditable comparison of five data mining models.")
    text(f"Gaussian Naive Bayes was chosen before opening test results, with selection Average Precision (AP) {selected.average_precision:.4f}. On the final 8,238-row test block, its AP is {winner.average_precision:.4f}, ROC-AUC {winner.roc_auc:.4f}, and Precision@824 {winner.precision_at_k:.2%}. The top 824 priorities contain 512 historical yes outcomes, a lift of {winner.lift_at_k:.2f} over the test block's base rate.")
    text("These results support retrospective ranking utility, not reliable future probabilities or a guaranteed commercial benefit. The yes prevalence rises from 4.81% in train to 30.83% in test. GaussianNB's threshold accuracy is only 46.77%, and its test Brier score is 0.2546. Ranking performance and calibration must therefore be assessed separately.")
    table(["Chapter", "Content"], [
        ["1 Introduction", "Case, stakeholder need, objectives, scope, and KDD learning outcomes"],
        ["2 Theoretical Basis", "Data concepts, five model families, calibration, and metric rationale"],
        ["3 Methods", "Data audit, pre-contact features, temporal splits, tuning, and frozen selection"],
        ["4 Implementation and Evaluation", "Architecture, model results, capacity policy, drift, interpretation, and software quality"],
        ["5 Conclusion and Suggestions", "Findings, limitations, and a responsible next experiment"],
        ["6 Appendix", "Public link targets, reproducibility, checksums, references, and publication status"],
    ], [135, 364.28])
    if not identity_complete or not publication_verified:
        text("Submission caveat: supply real cover identities and verify the public slide/data/source links before uploading. A public repository does not publish unpushed local files.")

    page("Chapter 1 Introduction", "Case study and objectives")
    heading("Business problem")
    text("A bank marketing team operates with a finite calling budget. Calling every eligible observation is not possible, while contact histories and customer attributes may contain useful patterns. DepoRank estimates the association between pre-contact attributes and historical term deposit subscription, then sorts observations for a capacity-constrained campaign list.")
    text("The stakeholder is a campaign analyst or supervisor. The core decision is how to allocate a fixed number of contacts, not whether a person qualifies for credit. A score is an estimated historical probability, a threshold label is a classification output, and a contact priority is a separate operational policy.")
    heading("Project objectives")
    text("The project implements the full KDD process, compares at least five distinct model families under the same feature/split protocol, justifies imbalance-aware and capacity-based metrics, and exposes validated single/batch scoring through a reproducible local app. Dataset provenance, configuration, model selection, and final-test evidence must remain traceable.")
    heading("Scope and success criteria")
    text("Success means that data and code can be inspected, the model comparison is leakage-aware, the final decision is fixed before test, capacity selects exactly K records, and the app produces understandable outputs with stable failure handling. Predictive scores need not be high for the experiment to be methodologically valid; weak results and drift must be reported rather than hidden.")
    table(["Learning outcome", "Project evidence"], [
        ["LO 1 Data and preprocessing concepts", "Dataset dictionary, audit, warehouse distinction, KDD stages, and leakage exclusions"],
        ["LO 2 Apply data mining techniques", "Five trained model families, temporal CV, calibration, metric comparison, and running application"],
        ["LO 3 Trends and research frontiers", "Temporal drift, frozen calibration, subgroup checks, responsible evaluation, and prospective monitoring recommendations"],
    ], [190, 309.28])

    page("Chapter 2 Theoretical Basis", "Data concepts and the KDD process")
    text("A dataset is an organized set of observations and attributes. Here, categorical fields describe customer/contact history and numeric fields encode age, earlier contact counts, and time since a previous campaign. The binary target records whether a term deposit subscription occurred. Data quality includes schema validity, categorical domains, consistency, missingness, duplicates, and provenance.")
    text("A data warehouse integrates historical data from operational sources for analysis, commonly with curated dimensions and facts. This prototype uses one public CSV and a local artifact registry; neither its raw folder nor its JSON manifests constitute an enterprise data warehouse. A warehouse is not necessary for the one-source academic experiment, although a production design could integrate permissioned customer snapshots, contact events, and outcomes.")
    table(["KDD stage", "Application in DepoRank"], [
        ["Data selection", "Choose the official full dataset and pre-contact core attributes; pin checksum and original order."],
        ["Preprocessing", "Parse and validate records; audit duplicate/unknown/inconsistent values; reject malformed input."],
        ["Transformation", "Separate the pdays sentinel from elapsed days; encode categories and scale numeric values inside training folds."],
        ["Data mining", "Search five classifier families using expanding-window CV, then calibrate frozen base models."],
        ["Interpretation and evaluation", "Select/lock before test; compare discrimination, calibration, and capacity utility; inspect importance and subgroups."],
    ], [125, 374.28])
    text("Leakage occurs when a model or preprocessing step receives information unavailable at the decision point or derived from future evaluation data. Fitting an encoder before CV, using post-call duration, tuning on final test, or choosing a new winner after viewing test would contaminate the evidence. The project's separate temporal blocks and pipeline fitting rules address these risks but do not eliminate missing as-of timestamps or unobserved customer overlap.")
    heading("Research context")
    text("Modern trustworthy evaluation includes out-of-time validation, probability calibration, data/version lineage, drift monitoring, and subgroup evidence. DepoRank demonstrates these concerns at prototype scale rather than claiming state-of-the-art banking performance. The related telemarketing study provides domain context, not a directly comparable accuracy benchmark for this stricter feature/split protocol [1, 2].")

    page("Chapter 2 Theoretical Basis", "Five model families and calibration")
    table(["Model", "Mechanism and limitations"], [
        ["Logistic Regression", "Models log-odds as a linear function of encoded features. Regularization controls coefficient size; it offers a compact baseline but misses nonlinear interactions unless features encode them."],
        ["Gaussian Naive Bayes", "Uses class-conditional Gaussian likelihoods and conditional independence. Fast and parsimonious, but correlated/one-hot features violate its simplifying assumptions; calibration is still required."],
        ["Decision Tree", "Recursively partitions feature space into rules. Captures nonlinear effects and interactions, but deep trees overfit and can produce unstable probability estimates."],
        ["Random Forest", "Averages many randomized decision trees. Reduces variance and captures interactions, but costs memory and does not guarantee robustness to temporal distribution shifts."],
        ["Gradient Boosting", "Builds additive trees sequentially to reduce a loss. Captures nonlinear relationships; learning rate, depth, and estimator count control complexity and fit cost."],
    ], [135, 364.28])
    text("All five families receive the same core raw features and chronological blocks. Their preprocessing stays inside the pipeline. A dummy prior model predicts the training class prevalence, while seeded random ranking tests a no-information ordering. Neither baseline counts as a sixth learned model family.")
    heading("Calibration")
    text("A model can rank observations well while estimating probabilities poorly. Sigmoid calibration learns a mapping from base scores to outcome likelihood on the independent calibration block. FrozenEstimator prevents recalibration from refitting the already trained base model. Calibration quality is then inspected using Brier loss and reliability bins, not assumed from the use of a calibrator [3].")
    text("Training class weighting is part of selected model configurations, not a replacement for independent calibration or minority-sensitive evaluation. No synthetic oversampling is added. Keeping chronology intact is more important than producing a superficially balanced random split.")
    heading("Temporal validity")
    text("The official full CSV is date-ordered, but individual exact decision timestamps and unique customer identifiers are not supplied. Preserving row order approximates an out-of-time experiment; it cannot prove strict customer-group separation or verify every field's as-of availability. Production validation would need both.")

    page("Chapter 2 Theoretical Basis", "Evaluation metrics and stakeholder relevance")
    table(["Metric", "Reason for inclusion"], [
        ["Average Precision", "Primary search/selection metric. Summarizes precision at recall increments, emphasizing positive-class retrieval without relying on a fixed threshold."],
        ["ROC-AUC", "Threshold-independent discrimination. Useful context, but may obscure weak precision when positive outcomes are uncommon."],
        ["Precision and Recall at K", "Directly answer whether limited campaign slots concentrate yes outcomes and how many positives remain outside capacity."],
        ["Lift at K", "Normalizes Precision@K by the evaluation block's base rate. Values above one indicate concentration above that block's average, not causal treatment uplift."],
        ["Brier and reliability bins", "Evaluate probability error and calibration. Low probability loss is distinct from high ranking utility."],
        ["F1 and confusion matrix", "Describe the classification decision at the selection-frozen threshold, including false positives/negatives."],
        ["Accuracy", "A supplementary threshold metric, always interpreted against class prevalence and the majority baseline."],
    ], [135, 364.28])
    text("Definitions: Precision@K = TP within K / K; Recall@K = TP within K / all positives; Lift@K = Precision@K / prevalence; Brier = mean((p_yes - y)^2). F1 is the harmonic mean of threshold precision and recall. AP follows average_precision_score, not trapezoidal integration [4].")
    text("At fraction q, K = ceil(q*N). Ranking uses descending p_yes and ascending input row ID for ties. A decision threshold is not a contact quota: changing q must not retune probabilities or classification thresholds. Undefined metrics are null with explicit reasons rather than fabricated values.")
    text("The choice of AP is tied to finding rare historical positives; Precision@K and Lift@K align with the campaign's fixed-capacity decision. This choice does not establish monetary ROI because contact cost, deposit margin, and causal incremental response are absent.")

    page("Chapter 3 Methods", "Dataset selection and audit")
    text("The selected source is UCI Bank Marketing, bank-additional-full, distributed with its official English dictionary. It contains 41,188 contact observations, 20 original inputs, and target y, from a Portuguese bank between May 2008 and November 2010. The full file is used rather than the randomly selected 4,119-row sample. Its date-ordered original positions support the project's chronological proxy split [1].")
    table(["Audit item", "Recorded result", "Treatment"], [
        ["Rows and columns", "41,188 and 21", "Enforce the expected variant and preserve original order."],
        ["Target", "4,640 yes; 36,548 no", "Map yes=1, no=0; no target enters features."],
        ["Exact duplicate rows", "12", "Record in audit; do not silently deduplicate."],
        ["Errors and warnings", "0 errors; 4,110 warnings", "Warnings flag inconsistencies, not automatic corrections."],
        ["Unknown default", "8,597", "Retain the explicit valid category."],
        ["Unknown education", "1,731", "Retain the explicit valid category."],
        ["Unknown housing and loan", "990 each", "Retain the explicit valid category."],
        ["Unknown job and marital", "330 and 80", "Retain the explicit valid category."],
    ], [145, 120, 234.28])
    text("The official CSV is kept unchanged. The dataset SHA-256 is checked before training, and its dictionary hash and source URL are included in the ingestion manifest. Duplicate rows can reflect recurring observations rather than safe-to-delete customer duplicates. No unique customer identifier is available, so customer-level grouping is a stated limitation.")
    text("At app input time, parsing and domain validation occur before scoring. Invalid CSV batches are rejected entirely with a downloadable row/field error report. Invalid values are never silently changed to unknown. Valid but inconsistent values receive warnings that remain visible to the operator.")
    text("Dataset and dictionary checksums are printed in Chapter 6; the complete raw-field dictionary accompanies the dataset submission ZIP.")

    page("Chapter 3 Methods", "Feature availability and transformation")
    table(["Core input", "Meaning and handling"], [
        ["age", "Integer age; application range 1-120."],
        ["job, marital, education", "Categorical profile attributes with explicit official categories and supported unknown values."],
        ["default, housing, loan", "Credit-default and loan indicators; categorical no/yes/unknown, not a credit approval decision."],
        ["pdays", "Previous-campaign elapsed days; 999 is represented by an availability flag and zero pdays_value."],
        ["previous", "Nonnegative previous-campaign contact count."],
        ["poutcome", "Previous campaign outcome: failure, nonexistent, or success."],
    ], [145, 354.28])
    text("The ten raw features become eleven intermediate fields because pdays is replaced by previously_contacted and pdays_value. When pdays=999, both values become zero; for pdays=0, previously_contacted remains one. One-hot encoding further expands categorical fields. The numeric scaler and encoder are fitted inside each training fold and are then reused without fitting on calibration, selection, or test.")
    heading("Excluded fields and leakage rationale")
    text("duration is excluded because the final call duration is unknown before the contact. campaign includes contacts in the current campaign, including the last contact, and is conservatively excluded from the pre-contact baseline. y and row_id are never features. Current-contact contact/month/day_of_week and all economic context fields are excluded from core_v1; exact point-in-time availability is not documented for a production use case.")
    text("The optional context_v1 definition remains an unrun retrospective extension. No comparison or performance claim is made for it. Expanding the completed baseline after observing test results would require a new version and independent holdout, not a cosmetic change to the existing feature manifest.")
    heading("Validity boundary")
    text("Removing obvious leakage does not prove all remaining inputs are historically point-in-time correct. A real deployment would need consented customer snapshots at the proposed calling time, event timestamps, and customer IDs. The report distinguishes this limitation from schema correctness.")

    page("Chapter 3 Methods", "Chronological validation and model search")
    table(["Block", "Original positions", "Rows", "Yes", "Prevalence"], [
        [name.title(), f"[{split['ranges'][name][0]}, {split['ranges'][name][1]})", f"{split['counts'][name]['rows']:,}", f"{split['counts'][name]['yes']:,}", f"{split['counts'][name]['prevalence']:.2%}"]
        for name in ("train", "calibration", "selection", "test")
    ], [85, 150, 80, 75, 109.28])
    text("No shuffling occurs. Original CSV positions are a temporal proxy rather than exact dates or unique customer identities. The fixed 60/10/10/20-style boundaries ensure full, disjoint row coverage. Duplicates and cross-block overlap are recorded in split_manifest.json; the limited duplicate overlap is not silently removed.")
    table(["Inner fold", "Fit positions", "Validation positions"], [
        ["1", "[0, 6178)", "[6178, 12356)"],
        ["2", "[0, 12356)", "[12356, 18534)"],
        ["3", "[0, 18534)", "[18534, 24712)"],
    ], [95, 190, 214.28])
    text("Each fold uses past training positions and a later validation segment, with gap=0 and both classes required. Grid search uses AP and raises errors instead of silently accepting failed fits. After search, each selected base model is fitted on the full train block, then left frozen for calibration.")
    table(["Family", "Grid candidates", "Parameters searched"], [
        ["Logistic Regression", "6", "C {0.1,1,10}; class_weight {None,balanced}"],
        ["GaussianNB", "3", "var_smoothing {1e-11,1e-9,1e-7}"],
        ["Decision Tree", "16", "depth {3,6,10,None}; leaf {1,20}; class_weight"],
        ["Random Forest", "8", "depth {8,None}; leaf {1,10}; class_weight; 200 trees fixed"],
        ["Gradient Boosting", "8", "trees {100,200}; rate {0.05,0.1}; depth {2,3}"],
    ], [135, 70, 294.28])
    text("Total: 41 candidates x 3 folds = 123 fold fits, excluding final base refits and calibration. Seed=42 and detailed fixed settings are in configs/experiment.yaml. The protocol includes no repeated independent split experiments or confidence intervals.")

    page("Chapter 3 Methods", "Calibration and frozen selection")
    text("Sigmoid calibration is fitted only on positions [24712,28831), using FrozenEstimator to preserve trained base estimators. Selection positions [28831,32950) then supply probability/ranking metrics, model comparison, and threshold selection. The final test block remains unused until the selection lock is written.")
    table(["Model", "Selection AP", "Precision at 412", "Brier", "Threshold"], [
        [MODEL_NAMES[model], f"{selection.loc[model,'average_precision']:.4f}", f"{selection.loc[model,'precision_at_k']:.4f}", f"{selection.loc[model,'brier_score']:.4f}", f"{lock['thresholds_by_model'][model]:.2f}"] for model in model_order
    ], [145, 90, 100, 80, 84.28])
    text("GaussianNB is the selection winner with AP 0.2318. The next strongest selection AP is Logistic Regression at 0.1773, well outside the practical tie interval of 0.005. Tie-breakers therefore do not alter this winner. They are nevertheless specified in advance: higher Precision@10%, lower Brier, lower median inference latency, and model ID.")
    text("Each model's decision threshold maximizes F1 on the selection block over 101 grid points from zero to one; a threshold tie prefers the higher threshold. GaussianNB's frozen threshold is 0.09. This relatively low classification threshold does not enforce a 10% contact quota; capacity is applied independently by ranking.")
    heading("Freeze and final evaluation")
    text("selection_lock.json pins the dataset/config/split hashes, selection metrics hash, all candidate model hashes, winner, feature set, thresholds, and policy version. The active run pointer references this locked winner. Final test evaluates all fixed candidates and baselines once, without choosing a new winner or refitting any model.")
    text("Changing the best-looking model after opening test results would turn test into another selection set. Even where another candidate has higher test Precision@K, the app continues to use GaussianNB. Saved-prediction reproduction checks equivalence; it does not authorize a second final-test run.")

    page("Chapter 4 Implementation and Evaluation", "Application architecture and workflow")
    table(["Layer", "Existing implementation"], [
        ["Entry point", "app.py configures Streamlit; src/bank_marketing/ui/pages.py renders seven English pages."],
        ["Data and features", "data/ingest.py and audit.py parse/validate; splits.py preserves chronology; feature transformers fit inside pipelines."],
        ["Machine learning", "ml/train.py, calibrate.py, selection.py, final.py, and registry.py manage training, freeze, evaluation, and verified loading."],
        ["Services", "services/scoring.py validates and predicts; ranking.py selects exact K; export.py emits sanitized CSV/JSON with provenance."],
        ["Storage and evidence", "Atomic writes, manifests, checksums, config_resolved.yaml, saved predictions, environment.json, and selection_lock.json."],
        ["Submission tooling", "A checksum-pinned compressed demo, safe restore script, English report/deck builders, and four-file packaging script."],
    ], [125, 374.28])
    text("The architecture separates an offline experiment runner from a serving application. The app consumes verified, frozen pipelines and evidence instead of running model search during interaction. The raw dataset, model parameters, winner, probabilities, thresholds, and integrity-protected files retain their recorded provenance.")
    heading("Input to output")
    text("A single-record form or UTF-8 CSV supplies ten core inputs. CSV upload supports comma/semicolon delimiters within 20 MB and 50,000 rows. Validation is all-or-nothing. Successful scoring produces p_yes and a fixed-threshold label; capacity K or fraction q then chooses exactly K highest scores. Ties use input row order. CSV/JSON downloads include model/run provenance, and CLI exports include a metadata sidecar.")
    text("Changing K/q only reranks cached scores. Changing data, mode, model identity, or threshold invalidates stale results. Invalid uploads clear old results. No training occurs on Streamlit reruns. Documentation remains accessible when the registry is empty or corrupt, while prediction is disabled.")
    text("Session cleanup removes uploaded inputs/results from session state. New CSV uploads never become model artifacts; unknown joblib/pickle uploads are not accepted. Trusted model loading is fail-closed on hash or environment mismatch. A cryptographic checksum detects modification but does not make an untrusted pickle safe.")

    page("Chapter 4 Implementation and Evaluation", "Model comparison on the frozen final test")
    text("All rows below use the same 8,238-observation final test block, the same core_v1 definition, and selection-frozen thresholds. Capacity q=0.10 produces K=824. These numbers come directly from the verified run's test_metrics.csv; rounding here is for readability only.")
    table(["Model", "AP", "ROC-AUC", "P at K", "R at K", "Lift", "Brier"], [
        [MODEL_NAMES[model], *[f"{test.loc[model,metric]:.4f}" for metric in ("average_precision", "roc_auc", "precision_at_k", "recall_at_k", "lift_at_k", "brier_score")]] for model in model_order
    ], [139, 60, 66, 60, 60, 54, 60.28])
    image("model_ap", "Figure 1 Average Precision across selection and final test blocks", 235)
    text("GaussianNB has the highest final-test AP (0.5076) and ROC-AUC (0.6667), consistent with its pre-test selection. Logistic Regression has better top-K precision (0.6517), while Gradient Boosting has the lowest learned-model Brier (0.2419). No model dominates all objectives; test findings do not trigger reselection.")
    text("Random Forest's test Lift@K is 0.9250, below the test base rate despite being a flexible ensemble. This is a valid observed result, not removed from the comparison. More complex models are not assumed to be more robust to temporal drift.")
    text("Diagnostic baselines: dummy prior has test AP 0.3083 and Brier 0.2810; seeded random ranking has AP 0.3094 and Lift@K 0.9879. Dummy probabilities tie, so deterministic row-order top-K metrics are not a randomized expected baseline. Random ranking has no calibrated-probability or classification metrics.")

    page("Chapter 4 Implementation and Evaluation", "Capacity results and classification tradeoffs")
    table(["GaussianNB at q 10 percent", "Observed test outcome"], [
        ["Evaluation population", "8,238 contact observations; 2,540 historical yes outcomes"],
        ["Capacity", "ceil(0.10 x 8,238) = 824 priorities"],
        ["Yes within priorities", "512 / 824 = 62.14% Precision@K"],
        ["No within priorities", "312 / 824 = 37.86% of priorities"],
        ["Yes outside priorities", "2,028; Recall@K = 512 / 2,540 = 20.16%"],
        ["Lift over block prevalence", "62.14% / 30.83% = 2.0153"],
    ], [210, 289.28])
    text("A finite quota necessarily leaves many positives outside the calling list. At q=10%, the model retrieves roughly one fifth of historical positives, not all of them. Increasing q changes this operational tradeoff but does not establish how many extra subscriptions calling would cause. An analyst must choose capacity from a real budget and evaluate a new prospective campaign.")
    heading("The frozen threshold is a different decision")
    table(["Actual class", "Predicted no", "Predicted yes"], [["no", "1,793", "3,905"], ["yes", "480", "2,060"]], [175, 162, 162.28])
    text("At threshold 0.09, precision is 34.53%, recall 81.10%, F1 0.4844, and accuracy 46.77%. The predicted-yes count is 5,965, far above the 824 contact quota. This illustrates why predicted_label and selected_for_contact must remain separate fields.")
    text("Always predicting no would have 69.17% test accuracy but recover no positives. This does not excuse the model's poor threshold accuracy; it explains why the report includes both ranking metrics and threshold errors instead of using one favorable number.")
    text("There is no observed deposit value, contact cost, or randomized no-contact control group. The 512 historical yes outcomes are not a forecast, profit estimate, or causal incremental response.")

    page("Chapter 4 Implementation and Evaluation", "Drift and probability calibration")
    image("prevalence", "Figure 2 Observed subscription prevalence changes across chronological blocks", 192)
    text("Target prevalence is 4.81% in train, 10.15% in calibration, 11.99% in selection, and 30.83% in test. This strong shift changes the meaning of a learned probability mapping and the behavior of a fixed threshold. AP values across blocks have different base rates and should not be interpreted as a pure improvement in model quality.")
    image("calibration", "Figure 3 GaussianNB reliability bins on selection and test using the frozen calibrator", 196)
    text("Calibration is a measured property, not guaranteed by applying sigmoid calibration. GaussianNB's test Brier score is 0.2546 versus 0.1050 on selection. The reliability plot shows that calibration does not transfer cleanly to the later block. The bins are descriptive estimates and no confidence intervals are claimed.")
    text("A fresh deployment should monitor feature distributions, outcome prevalence, calibration, and capacity metrics as labels arrive. Recalibrating on test now would contaminate this academic baseline. An approved update needs new calibration/selection data and an independent out-of-time holdout.")

    page("Chapter 4 Implementation and Evaluation", "Interpretation and subgroup evidence")
    image("importance", "Figure 4 Five-repeat permutation importance for the selection block only", 227)
    text("The strongest mean AP decreases after permutation are poutcome (0.0396), pdays (0.0327), and job (0.0239). These findings suggest that previous campaign history and employment category contribute to the fitted model's ranking on selection. The five permutation repeats quantify sensitivity within that block, not repeated independent model experiments.")
    text("loan has a slightly negative mean importance (-0.0002). Negative values are retained rather than hidden. Correlated attributes and the Naive Bayes independence assumption limit attribution: permutation importance is neither a causal explanation nor proof that modifying a customer's attribute would improve subscription.")
    heading("Subgroups and responsible interpretation")
    text("The recorded interpretation report checks age groups, job, marital status, and education on selection. It includes sample sizes, global-capacity selection rates, subgroup precision/recall, and calibration bins. Groups with fewer than 100 observations are explicitly marked as insufficient evidence. Small or correlated subgroup summaries do not establish fairness.")
    text("Customer profile attributes may reflect sensitive social differences. Before production use, a bank should define lawful/ethical use constraints, audit harmful disparities with sufficiently sized current data, and document mitigation decisions. DepoRank does not make credit or entitlement decisions and does not claim regulatory compliance.")
    text("Interpretation is confined to selection and does not change the feature set, winner, threshold, or test baseline. The complete machine-readable evidence is in the demo archive's interpretation.json.")

    page("Chapter 4 Implementation and Evaluation", "Software verification and local performance")
    if quality:
        text(f"The recorded software validation contains {quality['passed']} passing tests, {quality['failures']} failures, {quality['errors']} errors, and {quality['skipped']} skipped tests. Runtime was {quality['seconds']:.2f} seconds. The evidence summary is reports/software_quality.json. Software test success is not a claim about prediction quality or production readiness.")
    else:
        text("The pre-translation baseline passed 230 tests. Updated quality results must be produced by running the test suite; this report does not invent a later test count.")
    table(["Verification area", "Examples covered"], [
        ["Data contracts", "Malformed/ambiguous CSV, duplicate headers, invalid values, limits, warnings, and raw hashes."],
        ["Leakage and models", "Fold-specific encoders, sentinel handling, expanding-window validation, calibration freeze, thresholds, and model family definitions."],
        ["Registry and safety", "Lock/model hashes, active-pointer corruption, path escape, environment mismatch, and protected output paths."],
        ["App workflow", "Seven pages, single/batch input, exact K, reranking without retraining, stale-result invalidation, session cleanup, and errors."],
        ["Submission tooling", "Trusted demo restoration, checksum failure, safe extraction, non-overwrite policy, and English navigation labels."],
    ], [145, 354.28])
    table(["Recorded local measurement", "Value"], [
        ["Warm registry startup", f"{benchmark['startup_registry_warmed_seconds']:.3f} seconds"],
        ["Single scoring median over 20 repeats", f"{benchmark['single_median_seconds_20_repetitions']:.3f} seconds"],
        ["10,000-row parse/validate/predict/rank", f"{benchmark['batch_parse_validate_predict_rank_seconds']:.3f} seconds"],
        ["Sampled peak process RSS", f"{benchmark['peak_process_rss_bytes_sampled'] / 1024**2:.1f} MiB"],
    ], [330, 169.28])
    text("Measurements come from the original October 5, 2026 experiment on Python 3.11.16, Windows, 20 logical CPUs, and approximately 23.7 GiB RAM. Warm startup excludes cold Python/Streamlit startup. RSS polling every 5 ms is not complete allocation instrumentation. These are local observations, not cross-machine benchmarks or banking SLAs.")
    text("The source ZIP provides setup instructions, exact dependencies, tests, the unchanged dataset, and a compact trusted archive containing all five models. Restoration avoids retraining for a classroom demo and refuses to overwrite conflicting local artifacts. Final human checks still include the actual presentation computer, keyboard navigation, offline execution after setup, and no-login public downloads.")

    page("Chapter 5 Conclusion and Suggestions", "Findings and recommendations")
    heading("Conclusion")
    text("The completed project implements all five KDD stages and compares five distinct classifier families using a shared, leakage-aware temporal protocol. Gaussian Naive Bayes is the selection-frozen winner. On final test it reaches AP 0.5076 and concentrates 512 historical yes outcomes into 824 priorities, yielding 62.14% Precision@K and 2.02 lift against the later block's 30.83% base rate.")
    text("The evidence also contains important weaknesses. Probability calibration degrades under a large prevalence shift; the chosen classification threshold has poor accuracy; Random Forest's top-K lift is below one; and another model can outperform the winner on one test metric. These findings are disclosed rather than resolved by post-test tuning. The project demonstrates an auditable retrospective decision workflow, not a bank-ready or causal marketing system.")
    heading("Limitations")
    text("The source is historical Portuguese data, not current Indonesian customer data. Contact observations cannot be verified as unique customers. Exact as-of timestamps, consent records, real campaign costs/margins, and randomized treatment outcomes are absent. Only one fixed chronological experiment is reported; uncertainty intervals and external validation are not available. The optional context experiment is unrun.")
    heading("Suggested next experiment")
    text("First obtain authorized current data and define the real decision time, eligibility rules, customer grouping, costs, and capacity. Construct point-in-time features with customer IDs and new out-of-time holdouts. Retain a simple baseline, compare more models only under a predeclared search budget, and estimate uncertainty with an appropriate temporal procedure.")
    text("Next validate calibration and subgroup effects on sufficiently sized data. Monitor prevalence, score distributions, capacity precision, label delay, and calibration. Refresh calibration or retrain only through a versioned protocol with new independent evaluation data. Explain model changes to the operator and preserve lineage.")
    text("If the business question is incremental response to being contacted, use a controlled prospective campaign and an appropriate causal/uplift design. Classification on historical subscription labels alone cannot identify that effect. A production rollout requires business, privacy, ethics, and operational approval outside this academic project's scope.")

    page("Chapter 6 Appendix", "Public links and reproduction")
    repository = identity["repository_url"].rstrip("/")
    branch = identity["branch"]
    urls = {
        "slides": repository + f"/raw/refs/heads/{branch}/presentation/slides.pptx",
        "dataset": repository + f"/raw/refs/heads/{branch}/data/raw/bank-additional-full.csv",
        "source_code": repository,
    }
    if publication_verified:
        text("The following slide, dataset, and source links passed anonymous HTTP access checks recorded in reports/publication_checks.json. The team should also open and download them in a signed-out browser before submission.")
    else:
        text("The following publication targets are intended for anyone to download/view without login. The repository was confirmed public on October 6, 2026. Newly prepared slide/data/source files must still be pushed before their links can satisfy the guideline; local file existence does not prove public availability.")
    link("Presentation slides", urls["slides"])
    link("Project dataset", urls["dataset"])
    link("Source code repository", urls["source_code"])
    link("Official dataset download fallback", "https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip")
    link("Official dataset documentation", "https://archive.ics.uci.edu/dataset/222/bank+marketing")
    heading("Submission bundle")
    table(["File", "Purpose"], [
        ["DepoRank_Presentation.pptx", "Editable stakeholder presentation"],
        ["DepoRank_Dataset.zip", "Raw CSV, official dictionary, provenance, and attribution"],
        ["DepoRank_Source_Code.zip", "Runnable repository snapshot, tests, docs, data, and frozen demo"],
        ["DepoRank_Report.pdf", "This six-chapter report"],
    ], [215, 284.28])
    text("Reproduction on Python 3.11: install requirements.lock, install this project editable with no dependency resolution, run scripts/restore_demo.py, verify run 20261005T141242Z_4ac2cd71, then launch Streamlit with app.py. README.md includes exact Windows and Unix commands. The restore archive keeps locked evidence byte for byte; historical metadata may retain original language while the report/UI provide English explanations.")
    text("Before submission, add real member names/NIM, regenerate the artifacts, commit/push to the public main branch, and check all links signed out. scripts/check_public_links.py performs anonymous HTTP checks; a human must also open/download the results. The team separately handles proposal, demo video, and live presentation. No such activity is falsely recorded as completed.")

    page("Chapter 6 Appendix", "Evidence identifiers and references")
    text("Frozen run ID: " + run_id, "Caption")
    text("Dataset SHA-256: " + manifest["dataset_sha256"], "Caption")
    text("Resolved configuration object SHA-256: " + manifest["config_sha256"], "Caption")
    text("Selection-lock payload SHA-256: " + lock["lock_sha256"], "Caption")
    text("Configuration and lock identifiers use storage.object_hash: SHA-256 over UTF-8 canonical JSON with sorted keys and compact separators. The lock's own lock_sha256 field is excluded. These are not raw YAML/JSON file checksums; file-level checksums remain in run_manifest.json.")
    text("Machine-readable evidence includes selection_metrics.csv, test_metrics.csv, model metadata, saved predictions, evaluation curves, split_manifest.json, environment.json, benchmark.json, and interpretation.json in the restored run. Hash-locked evidence is not edited for translation. Generated charts and report text can be refreshed without changing the locked model decision.")
    heading("References")
    references = [
        ("[1] UCI Machine Learning Repository. Bank Marketing dataset and official bank-additional-names dictionary. Creators: Sergio Moro, Paulo Cortez, and Paulo Rita. Dataset distributed under CC BY 4.0.", "https://archive.ics.uci.edu/dataset/222/bank+marketing"),
        ("[2] Moro, S., Cortez, P., and Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, 62, 22-31. DOI 10.1016/j.dss.2014.03.001.", "https://doi.org/10.1016/j.dss.2014.03.001"),
        ("[3] scikit-learn. Probability calibration and FrozenEstimator documentation. Used to define separate calibration of already fitted estimators.", "https://scikit-learn.org/stable/modules/calibration.html"),
        ("[4] scikit-learn. Model evaluation documentation, including Average Precision, ROC-AUC, classification metrics, and Brier loss.", "https://scikit-learn.org/stable/modules/model_evaluation.html"),
        ("[5] scikit-learn. TimeSeriesSplit documentation. Used for expanding-window validation restricted to the train block.", "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html"),
    ]
    for reference, url in references:
        text(reference)
        link("Source", url)
    text("Course source: user-supplied Data Mining final project guideline.docx, COMP6140001, academic year 2026/2027. Requirements applied: full KDD, at least five models, justified metrics, six report chapters, cover identities, and public slides/dataset/source links. The original guideline is preserved locally and excluded from public source packaging.", "Caption")

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D6E2E6"))
        canvas.line(48, 41, A4[0] - 48, 41)
        canvas.setFillColor(colors.HexColor("#536779"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(48, 27, "DepoRank | COMP6140001 Data Mining")
        canvas.drawRightString(A4[0] - 48, 27, str(document.page))
        canvas.restoreState()

    path = output / "final_report.pdf"
    document = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=48, rightMargin=48, topMargin=46, bottomMargin=58, title="DepoRank Term Deposit Campaign Prioritization", author="; ".join(member["full_name"] for member in identity.get("members", [])) or "Project team identity pending")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    (output / "final_report.md").write_text("\n".join(markdown), encoding="utf-8")
    summary = {
        "run_id": run_id, "winner_model_id": lock["winner_model_id"], "feature_set": lock["feature_set"],
        "selection_lock_sha256": lock["lock_sha256"], "status": manifest["status"],
        "identity_complete": identity_complete, "public_links_verified": publication_verified,
        "academic_status": "ready" if identity_complete and publication_verified else "pending_team_identity_and_publication",
        "test_metrics": winner.to_dict(), "software_quality": quality,
        "report": "reports/final_report.pdf", "slides": "presentation/slides.pptx", "public_links": urls,
    }
    write_json(output / "experiment_summary.json", summary)
    presentation_data = {
        "run_id": run_id, "winner": "Gaussian Naive Bayes", "threshold": 0.09,
        "models": [{"id": model, "name": MODEL_NAMES[model], "selection_ap": float(selection.loc[model, "average_precision"]), "test": {key: float(test.loc[model, key]) for key in ("average_precision", "roc_auc", "precision_at_k", "recall_at_k", "lift_at_k", "brier_score")}} for model in model_order],
        "splits": split["counts"], "importance": interpretation["permutation_importance"],
        "quality": quality, "members": identity.get("members", []), "links": urls,
    }
    write_json(output / "presentation_data.json", presentation_data)
    return summary

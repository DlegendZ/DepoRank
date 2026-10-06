import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { FileBlob, Presentation, PresentationFile } from "@oai/artifact-tool";

const project = path.resolve(process.env.PROJECT_ROOT ?? process.cwd());
const skill = process.env.PRESENTATIONS_SKILL_DIR;
const python = process.env.RUNTIME_PYTHON;
if (!skill || !python) throw new Error("Set PRESENTATIONS_SKILL_DIR and RUNTIME_PYTHON; see scripts/README.md.");
const { finalizePresentation, applyPresentationChartFont } = await import(pathToFileURL(path.join(skill, "container_tools/artifact_tool_utils.mjs")).href);
const build = path.join(project, ".build", "slides");
const output = path.join(project, "presentation");
await fs.mkdir(build, { recursive: true });
await fs.mkdir(output, { recursive: true });
const evidence = JSON.parse(await fs.readFile(path.join(project, "reports/presentation_data.json"), "utf8"));
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const NAVY = "#172D43";
const TEAL = "#087F83";
const GOLD = "#CB9344";
const MUTED = "#536779";
const WHITE = "#FFFFFF";
const FONT = "Arial";
const notesSource = `Evidence: frozen run ${evidence.run_id}; reports/presentation_data.json; selection/test metrics and selection_lock.json. Chart values are rounded to six decimal places; full precision remains in the frozen evidence. Dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing; Moro et al. (2014), https://doi.org/10.1016/j.dss.2014.03.001.`;

function text(slide, value, left, top, width, height, size = 26, color = NAVY, bold = false) {
  const shape = slide.shapes.add({ geometry: "textbox", position: { left, top, width, height }, fill: "none", line: { fill: "none", width: 0 } });
  shape.text = value;
  shape.text.style = { typeface: FONT, fontSize: size, color, bold, autoFit: "none" };
  return shape;
}

function rectangle(slide, left, top, width, height, fill) {
  return slide.shapes.add({ geometry: "rect", position: { left, top, width, height }, fill, line: { fill: "none", width: 0 } });
}

function page(title, subtitle = "", notes = "") {
  const slide = presentation.slides.add();
  slide.background.fill = WHITE;
  text(slide, title, 64, 46, 1152, 70, 40, NAVY, true);
  if (subtitle) text(slide, subtitle, 66, 125, 1148, 60, 23, MUTED);
  rectangle(slide, 64, 658, 1152, 2, "#D6E2E6");
  text(slide, "DepoRank | COMP6140001 Data Mining", 64, 674, 930, 24, 15, MUTED);
  text(slide, String(presentation.slides.items.length).padStart(2, "0"), 1150, 674, 66, 24, 15, MUTED);
  slide.speakerNotes.textFrame.setText(`${notes}\n\n${notesSource}`);
  return slide;
}

function table(slide, values, widths, top = 215, height = 350, size = 22) {
  const result = slide.tables.add({ rows: values.length, columns: values[0].length, left: 64, top, width: 1152, height, values, columnWidths: widths });
  for (let row = 0; row < values.length; row += 1) {
    for (let column = 0; column < values[0].length; column += 1) {
      const cell = result.getCell(row, column);
      cell.fill = row === 0 ? NAVY : row % 2 ? "#F0F5F6" : WHITE;
      cell.text.style = { typeface: FONT, fontSize: size, color: row === 0 ? WHITE : NAVY, bold: row === 0 };
    }
    result.rows[row].height = height / values.length;
  }
  return result;
}

function chart(slide, categories, series, position, direction = "column", percentage = false) {
  const chartSeries = series.map(item => ({ ...item, values: item.values.map(value => Number(value.toFixed(6))) }));
  const result = slide.charts.add("bar", {
    position, categories, series: chartSeries, hasLegend: series.length > 1,
    legend: { position: "bottom", textStyle: { typeface: FONT, fontSize: 19, fill: NAVY } },
    barOptions: { direction, grouping: "clustered", gapWidth: 85 },
    xAxis: { numberFormatCode: direction === "bar" ? (percentage ? "0%" : "0.000") : undefined, textStyle: { typeface: FONT, fontSize: 19, fill: NAVY } },
    yAxis: { numberFormatCode: percentage ? "0%" : direction === "bar" ? "0.000" : "0.00", textStyle: { typeface: FONT, fontSize: 19, fill: NAVY } },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: percentage ? "0.0%" : direction === "bar" ? "0.0000;-0.0000" : "0.000", textStyle: { typeface: FONT, fontSize: 18, fill: NAVY } },
  });
  applyPresentationChartFont(result, { fontFamily: FONT });
  return result;
}

const cover = presentation.slides.add();
cover.background.fill = NAVY;
rectangle(cover, 72, 104, 92, 8, GOLD);
text(cover, "DepoRank", 68, 162, 1144, 122, 88, WHITE, true);
text(cover, "Term deposit campaign prioritization", 72, 318, 1080, 74, 39, WHITE);
text(cover, "Five-model comparison and a frozen contact-ranking application", 74, 428, 1050, 76, 26, "#BFD7DD");
text(cover, "COMP6140001 Data Mining\nBINUS University | Academic Year 2026/2027", 74, 568, 1050, 76, 22, "#BFD7DD");
cover.speakerNotes.textFrame.setText(`Introduce the stakeholder problem: finite contact capacity. Explain that the project compares five models and exposes an auditable local application. The presentation uses a fixed historical experiment, not prospective banking results.\n\n${notesSource}`);

let slide = page("A fixed quota needs ranking, not a yes label", "Retrospective test evidence for the selection-frozen GaussianNB model", "Lead with the campaign decision. At q=10%, K is ceil(0.10*8238)=824. The top 824 contain 512 yes and 312 no observations. Lift is relative to the test base rate, not causal marketing uplift. This is evidence of concentration, not guaranteed conversion or profit.");
text(slide, "824", 66, 223, 335, 103, 79, TEAL, true);
text(slide, "contact priorities", 70, 341, 325, 66, 26);
text(slide, "512", 446, 223, 335, 103, 79, TEAL, true);
text(slide, "historical yes outcomes", 450, 341, 337, 85, 26);
text(slide, "2.02", 826, 223, 335, 103, 79, GOLD, true);
text(slide, "lift over test prevalence", 830, 341, 350, 85, 26);
text(slide, "62.14% Precision@K | 20.16% Recall@K | 8,238 test observations", 70, 486, 1136, 60, 28, NAVY, true);
text(slide, "Historical outcomes are not causal uplift or a forecast of future subscriptions.", 70, 574, 1136, 45, 22, MUTED);

slide = page("Campaign planning within limited capacity", "Stakeholder: a campaign analyst allocating a fixed calling budget", "Describe the decision before discussing algorithms. A threshold classification and a ranked contact list solve different tasks. The app never decides credit eligibility, consent, or production campaign policy. One observation is a historical contact record, not a verified unique customer.");
text(slide, "Business question", 66, 215, 505, 48, 30, TEAL, true);
text(slide, "Which contact observations should receive the available campaign slots?", 66, 286, 505, 148, 32);
text(slide, "Application output", 667, 215, 547, 48, 30, TEAL, true);
text(slide, "A validated score, a fixed-threshold label, and exactly K contact priorities.", 667, 286, 547, 148, 32);
text(slide, "Measure retrieval and capacity utility. Do not claim ROI or incremental response.", 66, 506, 1148, 85, 27, NAVY, true);

slide = page("The dataset is historical and imbalanced", "UCI bank-additional-full | Portuguese bank contacts, May 2008-November 2010", "The unchanged official CSV contains 41,188 observations, 20 original inputs, and y. There are 4,640 yes and 36,548 no labels, 12 full-row duplicates, and 4,110 audit warnings. Duplicate/unknown values are recorded rather than silently repaired. The rows are date-ordered but no unique customer ID or exact decision timestamp is available.");
text(slide, "41,188", 68, 220, 515, 110, 72, TEAL, true);
text(slide, "contact observations", 70, 340, 510, 55, 28);
text(slide, "20 inputs + binary target y\n12 duplicates retained\nUnknown categories preserved", 70, 446, 510, 150, 26);
const datasetChart = slide.charts.add("doughnut", { position: { left: 652, top: 217, width: 534, height: 356 }, categories: ["yes", "no"], series: [{ name: "Target", values: [4640, 36548], points: [{ idx: 0, fill: TEAL }, { idx: 1, fill: NAVY }] }], hasLegend: true, legend: { position: "bottom", textStyle: { typeface: FONT, fontSize: 20, fill: NAVY } }, doughnutOptions: { holeSize: 62 }, dataLabels: { showValue: false, showCategoryName: false, showPercent: false } });
applyPresentationChartFont(datasetChart, { fontFamily: FONT });
text(slide, "yes 11.27% | no 88.73%", 650, 590, 548, 38, 23, MUTED);

slide = page("KDD turns raw records into auditable decisions", "All five required stages are present in the application and report", "Walk through selection, preprocessing, transformation, data mining, and interpretation/evaluation. Explain checksum verification, unknown-category preservation, pdays sentinel handling, fold-fitted encoders/scalers, model search, and frozen evaluation. A local CSV/registry is not an enterprise data warehouse.");
const stages = [
  ["Selection", "Official full CSV\nCore attributes\nChecksum pin"],
  ["Preprocessing", "Validate schema\nAudit duplicates\nReject bad rows"],
  ["Transformation", "pdays sentinel\nOne-hot encoding\nTrain-only fit"],
  ["Data mining", "Five families\nTemporal CV\nCalibration"],
  ["Evaluation", "Selection lock\nFinal test\nCapacity metrics"],
];
stages.forEach(([title, body], index) => {
  const left = 64 + index * 234;
  rectangle(slide, left, 239, 215, 8, index === 4 ? GOLD : TEAL);
  text(slide, String(index + 1), left, 278, 215, 74, 48, TEAL, true);
  text(slide, title, left, 371, 218, 55, 24, NAVY, true);
  text(slide, body, left, 443, 215, 143, 21, MUTED);
});

slide = page("Prevalence changes across four ordered blocks", "Original CSV positions preserved | No shuffling or refitting on future blocks", "Train has 24,712 rows and 4.81% yes; calibration and selection each have 4,119 rows with 10.15% and 11.99% yes; test has 8,238 rows and 30.83% yes. Original order is a temporal proxy, not proof of customer separation. This drift is a central limitation, not a post-test justification for changing the model.");
const blockNames = ["train", "calibration", "selection", "test"];
chart(slide, blockNames.map(name => `${name}\n${evidence.splits[name].rows.toLocaleString("en-US")} rows`), [{ name: "Yes prevalence", values: blockNames.map(name => evidence.splits[name].prevalence), fill: TEAL }], { left: 85, top: 205, width: 1090, height: 375 }, "column", true);
text(slide, "Train: fit/search | Calibration: probability mapping | Selection: decisions | Test: final evidence", 66, 595, 1150, 38, 21, MUTED);

slide = page("Five distinct model families share one protocol", "Same raw core features, chronological blocks, and evaluation definitions", "These are the five required model families, not five hyperparameter variants of one model. Dummy prior and seeded random ranking are extra baselines. Complexity does not guarantee better performance under temporal shift. All pipelines fit preprocessing only on train/fold data.");
table(slide, [
  ["Model family", "Mechanism", "Key tradeoff"],
  ["Logistic Regression", "Linear log-odds", "Compact; limited nonlinear interactions"],
  ["Gaussian Naive Bayes", "Class-conditional likelihoods", "Fast; conditional independence assumption"],
  ["Decision Tree", "Recursive rules", "Nonlinear; unstable when overfit"],
  ["Random Forest", "Averaged randomized trees", "Lower variance; higher memory cost"],
  ["Gradient Boosting", "Sequential additive trees", "Flexible; tuning and drift sensitivity"],
], [306, 342, 504], 214, 363, 23);
text(slide, "Dummy prior and random ranking are diagnostic baselines, not extra learned families.", 66, 601, 1148, 37, 21, MUTED);

slide = page("Keep training, calibration, and selection separate", "10 core features | 41 candidates | 3 expanding-window folds | 123 fold fits", "Core features are age, job, marital, education, default, housing, loan, pdays, previous, and poutcome. Exclude post-call duration, ambiguous current-campaign count, target, IDs, and current-contact/economic context. pdays=999 becomes a no-contact flag; pdays=0 still means previously contacted. Search AP inside train. Freeze base models for sigmoid calibration. Choose winner and F1 threshold on selection. Lock before opening test.");
text(slide, "Leakage controls", 66, 219, 518, 48, 31, TEAL, true);
text(slide, "Exclude duration and campaign\nNever fit preprocessing on future data\nHandle pdays=999 as a sentinel\nKeep row IDs out of the model", 66, 302, 518, 241, 28);
text(slide, "Decision controls", 670, 219, 542, 48, 31, TEAL, true);
text(slide, "FrozenEstimator sigmoid calibration\nSelection AP chooses the winner\nSelection F1 chooses thresholds\nFinal test cannot select a new winner", 670, 302, 542, 241, 28);
text(slide, "A threshold label does not determine the top-K contact quota.", 66, 591, 1144, 42, 25, NAVY, true);

slide = page("GaussianNB wins before test results are opened", "Selection AP is the declared primary metric | GaussianNB threshold fixed at 0.09", "Selection AP is 0.2318 for GaussianNB and 0.1773 for the next-best Logistic Regression. The gap exceeds the declared practical tie window 0.005, so no tie-breaker changes the outcome. In a tie, use Precision@10%, Brier, prediction latency, and model ID. The lock pins model hashes and decisions before final test. Do not reselect based on a nicer test metric.");
chart(slide, ["Logistic\nRegression", "Gaussian\nNaive Bayes", "Decision\nTree", "Random\nForest", "Gradient\nBoosting"], [{ name: "Selection AP", values: evidence.models.map(model => model.selection_ap), fill: TEAL }], { left: 83, top: 213, width: 1099, height: 351 });
text(slide, "The selection lock pins the winner, thresholds, models, and dataset/config hashes.", 66, 598, 1148, 37, 23, MUTED);

slide = page("Final test results show different tradeoffs", "8,238 observations | K=824 | Thresholds fixed on selection | No model reselection", "Read AP alongside capacity and calibration. GaussianNB has best test AP and ROC-AUC. Logistic Regression has highest Precision@K. Gradient Boosting has lowest learned-model Brier. Random Forest has lift below one. All five results are retained. AP differs across blocks partly because prevalence differs. No confidence interval or repeated independent split is claimed.");
table(slide, [
  ["Model", "AP", "ROC-AUC", "P@K", "R@K", "Lift@K", "Brier"],
  ...evidence.models.map(model => [model.name, ...["average_precision", "roc_auc", "precision_at_k", "recall_at_k", "lift_at_k", "brier_score"].map(key => model.test[key].toFixed(4))]),
], [306, 141, 141, 141, 141, 141, 141], 216, 351, 21);
text(slide, "GaussianNB remains active even though Logistic Regression has higher test Precision@K.", 66, 593, 1148, 44, 22, MUTED);

slide = page("Capacity and classification answer different questions", "A 10% contact quota is not the same as a probability threshold", "At K=824, GaussianNB selects 512 positives and 312 negatives; 2,028 positives are outside capacity. At threshold 0.09, the model labels 5,965 observations yes, with precision 34.53%, recall 81.10%, F1 0.4844, and accuracy 46.77%. Majority-no accuracy is 69.17%. This poor classification/calibration must be acknowledged even though ranking has utility.");
text(slide, "Contact quota", 66, 218, 517, 54, 32, TEAL, true);
text(slide, "824 selected for contact\n512 yes within capacity\n312 no within capacity\n2,028 yes outside capacity", 66, 305, 520, 256, 31);
text(slide, "Frozen threshold 0.09", 668, 218, 546, 54, 32, TEAL, true);
text(slide, "5,965 predicted yes labels\n34.53% precision\n81.10% recall\n46.77% accuracy", 668, 305, 545, 256, 31);
text(slide, "No costs, deposit values, consent, or randomized no-contact outcomes are available.", 66, 597, 1148, 40, 21, MUTED);

slide = page("Good ranking does not guarantee reliable probabilities", "Test Brier 0.2546 | Large prevalence shift | Production transfer is unproven", "Use this slide to make the strongest caveat clear. The calibrator was fit at roughly 10.15% prevalence and the later test block has 30.83% positives. The report's reliability plot documents transfer problems. Probability estimates and threshold labels need new validation before operational use. Do not recalibrate on test or call the current model production-ready.");
text(slide, "4.81%", 67, 237, 509, 103, 74, TEAL, true);
text(slide, "train yes prevalence", 70, 355, 505, 65, 29);
text(slide, "30.83%", 671, 237, 540, 103, 74, GOLD, true);
text(slide, "test yes prevalence", 674, 355, 536, 65, 29);
text(slide, "Validate calibration on new data. Monitor drift, label delay, and capacity performance.", 70, 477, 1134, 121, 32, NAVY, true);

slide = page("Previous campaign history matters on selection", "Permutation importance | Five repeats | Association, not causal explanation", "The strongest AP decreases are poutcome 0.0396, pdays 0.0327, and job 0.0239. Keep the small negative loan importance instead of hiding it. The repeats are sensitivity runs inside selection, not independent model experiments. Correlated features limit interpretation; subgroup sizes below 100 are marked insufficient evidence. Fairness is not established.");
const importance = [...evidence.importance].sort((first, second) => second.mean_ap_decrease - first.mean_ap_decrease);
chart(slide, importance.map(item => item.feature), [{ name: "Mean selection AP decrease", values: importance.map(item => item.mean_ap_decrease), fill: TEAL }], { left: 92, top: 210, width: 1070, height: 350 }, "bar");
text(slide, "Mean selection AP decrease; loan retains its negative importance (-0.0002).", 66, 562, 1148, 33, 19, MUTED);
text(slide, "Correlated features and small subgroups limit attribution. No fairness guarantee is made.", 66, 598, 1148, 38, 22, MUTED);

slide = page("The application demonstrates input through export", "Local scoring with a trusted frozen model | No training during page reruns", "Suggested live demo: open Overview and Data/KDD, compare selection/test metrics, predict one synthetic record, score demo_valid.csv at q=10%, change capacity without rescoring, export CSV/JSON metadata, then show whole-batch rejection using demo_invalid.csv. Clear session data afterward. The supplied labeled public subset is demonstration data, not a fresh holdout. Real customer data and model-file uploads must not be used.");
const workflow = [
  ["01", "Inspect", "KDD and model evidence"],
  ["02", "Predict", "One synthetic observation"],
  ["03", "Score", "Valid CSV at q=10%"],
  ["04", "Rerank", "Change K without training"],
  ["05", "Export", "CSV/JSON and provenance"],
  ["06", "Reject", "Invalid batch and error report"],
];
workflow.forEach(([number, title, detail], index) => {
  const left = 66 + (index % 3) * 390;
  const top = 227 + Math.floor(index / 3) * 185;
  text(slide, number, left, top, 80, 62, 36, TEAL, true);
  text(slide, title, left + 84, top + 3, 285, 51, 28, NAVY, true);
  text(slide, detail, left + 84, top + 67, 285, 85, 22, MUTED);
});
text(slide, `Software quality: ${evidence.quality.passed ?? "baseline 230"} passing tests; predictive results remain separate.`, 66, 601, 1148, 36, 21, MUTED);

slide = page("A next experiment must use current, permissioned data", "Historical Portugal results do not establish production validity in Indonesia", "Summarize the limitations: old data, absent unique customer IDs/as-of timestamps, no consent/cost/causal treatment evidence, strong drift, and one fixed experiment without uncertainty intervals. Recommend a new out-of-time/customer-grouped experiment, calibration/subgroup monitoring, and a randomized campaign if the question is incremental response. Changes need a new independent holdout.");
text(slide, "Current limitations", 66, 219, 515, 49, 31, TEAL, true);
text(slide, "Old and geographically different data\nNo verified customer/time isolation\nNo ROI or treatment-effect evidence\nCalibration and fairness unproven", 66, 306, 521, 255, 27);
text(slide, "Next validation", 670, 219, 542, 49, 31, TEAL, true);
text(slide, "Point-in-time customer snapshots\nNew out-of-time holdout\nDrift and subgroup monitoring\nProspective controlled campaign", 670, 306, 542, 255, 27);
text(slide, "Preserve the current frozen evidence. New decisions require a new evaluation protocol.", 66, 599, 1148, 38, 22, MUTED);

slide = page("Approve a validation study, not a production launch", "Retrospective ranking is promising; future probability reliability remains unproven", "Close with the stakeholder decision, not submission administration. Ask for a scoped validation study using current permissioned data, staffing capacity, lawful campaign eligibility, and a prospective controlled response experiment. The quoted metrics come from the frozen historical test only. Reproducible source and evidence will be available at the repository after the prepared artifacts are published; do not claim deployment or campaign approval has happened.");
text(slide, "Measured on the frozen test", 66, 224, 521, 49, 31, TEAL, true);
text(slide, "62.14% Precision@824\n20.16% Recall@824\n2.02 lift over test prevalence\n46.77% threshold accuracy", 66, 311, 521, 243, 28);
text(slide, "Decision requested", 670, 224, 542, 49, 31, TEAL, true);
text(slide, "Approve access to current data\nDefine capacity and contact eligibility\nTest calibration before probability claims\nMeasure response in a controlled study", 670, 311, 542, 243, 27);
text(slide, "Reproducible application and evidence: github.com/DlegendZ/DepoRank", 66, 600, 1148, 35, 21, MUTED);

const buildTag = Date.now();
const candidate = path.join(build, `candidate-${buildTag}.pptx`);
await (await PresentationFile.exportPptx(presentation)).save(candidate);
const validationOutput = path.join(project, ".build", "validated-slides");
await fs.mkdir(validationOutput, { recursive: true });
const finalPath = path.join(validationOutput, `validated-${buildTag}.pptx`);
const receipt = path.join(build, `validation-${buildTag}.json`);
const result = await finalizePresentation({
  workspaceDir: project, candidatePath: candidate, finalPath, pythonExecutable: python,
  integrityValidatorPath: path.join(skill, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skill, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit", "--require-native-table-slide", "7", "--require-native-table-slide", "10"],
  explicitTotalSlideCount: 16,
  requiredNativeTableOwnerSlides: [7, 10],
  requiredNativeChartOwnerSlides: [4, 6, 9, 13],
  materializeLiteralChartWorkbooks: true,
  fontPolicy: { basis: "design", families: [FONT] },
  verifyArtifactToolImport: true,
  receiptPath: receipt,
});
await fs.copyFile(finalPath, path.join(output, "slides.pptx"));
await fs.copyFile(receipt, path.join(build, "validation.json"));
const checkedPresentation = await PresentationFile.importPptx(await FileBlob.load(finalPath));
for (let index = 0; index < checkedPresentation.slides.items.length; index += 1) {
  const current = checkedPresentation.slides.items[index];
  const image = await checkedPresentation.export({ slide: current, format: "png", scale: 1.5 });
  await fs.writeFile(path.join(build, `slide-${String(index + 1).padStart(2, "0")}.png`), new Uint8Array(await image.arrayBuffer()));
  const layout = await current.export({ format: "layout" });
  await fs.writeFile(path.join(build, `slide-${String(index + 1).padStart(2, "0")}.layout.json`), await layout.text());
}
console.log(JSON.stringify({ slides: presentation.slides.items.length, file: path.join(output, "slides.pptx"), result }, null, 2));

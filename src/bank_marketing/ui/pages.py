from __future__ import annotations

import hashlib
import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from bank_marketing.data.audit import audit_dataset
from bank_marketing.data.ingest import read_csv
from bank_marketing.ml.evaluate import evaluate_scores
from bank_marketing.ml.registry import Registry
from bank_marketing.schemas import CATEGORIES, CORE_FEATURES, DomainError
from bank_marketing.services.export import export_errors, export_metadata, export_results
from bank_marketing.services.ranking import rank_records
from bank_marketing.services.scoring import score_records
from bank_marketing.storage import canonical_bytes, read_json

PAGES = ("Overview", "Data and KDD", "Model Comparison", "Single Prediction", "Batch Priorities", "Evaluation and Interpretation", "Documentation")
LOG = logging.getLogger(__name__)
MESSAGES = {
    "MODEL_NOT_READY": "Model not ready. Complete the offline freeze and activate stages; documentation remains available.",
    "ARTIFACT_INTEGRITY_ERROR": "Artifact integrity failed. Prediction stopped; check the registry using CLI verify.",
    "ENVIRONMENT_MISMATCH": "Environment mismatch. Use the Python version and requirements.lock recorded for this run.",
    "SCHEMA_INVALID": "Schema mismatch. Check headers, CSV format, required columns, and input mode.",
    "ROW_INVALID": "Invalid rows. The entire batch was rejected; correct fields listed in the error report.",
    "CAPACITY_INVALID": "Invalid capacity. Specify one K or q value within batch bounds.",
}


def show_error(error):
    st.error(f"{error.code}: {MESSAGES.get(error.code, 'Operation failed. Contact the maintainer with the error ID.')} ID: {error.error_id}")
    LOG.warning("state=error code=%s error_id=%s", error.code, error.error_id)
    if isinstance(error.details, dict) and error.details.get("errors"):
        st.dataframe(pd.DataFrame(error.details["errors"]), hide_index=True)
        st.download_button("Download error report", export_errors(error.details), "validation_errors.csv", "text/csv")


def _plot(frame, x, y, title, kind="line", color=None):
    if frame.empty:
        st.info("N/A: chart data unavailable.")
        return
    factory = px.bar if kind == "bar" else px.line
    fig = factory(frame, x=x, y=y, color=color, title=title, color_discrete_sequence=["#087F83", "#CB9344", "#172D43"])
    fig.update_layout(font_family="Georgia", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=15, r=15, t=55, b=20))
    st.plotly_chart(fig, width="stretch")


def _download(path, label=None):
    if path.is_file():
        st.download_button(label or path.name, path.read_bytes(), path.name, key=str(path))


def overview(root, run, ref):
    st.caption("01 / CAMPAIGN INTELLIGENCE")
    st.title("Prioritize contacts within campaign capacity")
    st.write("DepoRank ranks term deposit campaign contact observations. Local analysis, auditable model decisions, and no training during the demo.")
    if run is None:
        st.info("Empty: no locked run. Open Documentation for setup.")
        return
    manifest = read_json(run / "run_manifest.json")
    split = read_json(run / "split_manifest.json")
    audit = read_json(run / "audit.json")
    selection = pd.read_csv(run / "selection_metrics.csv")
    winner = selection[selection.model_id == ref.model_id].iloc[0]
    columns = st.columns(4)
    columns[0].metric("Public observations", f"{audit['counts']['rows']:,}")
    columns[1].metric("Target yes / full dataset", f"{audit['counts']['target']['yes'] / audit['counts']['rows']:.2%}")
    columns[2].metric("Model chosen on selection", ref.model_id)
    columns[3].metric("Selection AP", f"{winner.average_precision:.4f}")
    st.success(f"{manifest['status']} · {ref.run_id} · {ref.feature_set} · threshold {ref.threshold:.2f} · 5 model families")
    st.subheader("Four data blocks and one locked decision")
    frame = pd.DataFrame([{ "split": name, **values } for name, values in split["counts"].items()])
    _plot(frame, "split", "prevalence", "Target prevalence changes across blocks", "bar")
    st.dataframe(frame, hide_index=True)
    if (run / "test_metrics.csv").exists():
        st.subheader("Final test / no model reselection")
        test = pd.read_csv(run / "test_metrics.csv")
        selected = test[test.model_id == ref.model_id].iloc[0]
        a, b, c = st.columns(3)
        a.metric("Test AP", f"{selected.average_precision:.4f}")
        b.metric(f"Precision@{int(selected.capacity_k)}", f"{selected.precision_at_k:.2%}")
        c.metric(f"Lift@{int(selected.capacity_k)}", f"{selected.lift_at_k:.2f}")
    st.warning("Portugal 2008-2010; not validated for production in Indonesia. Probabilities are not guarantees, ranking is not causal uplift, and observations are not unique customers.")
    st.markdown("Source: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing). License: CC BY 4.0.")


def data_kdd(root, run, ref):
    st.title("Data and KDD")
    st.write("Public raw data is read-only. Original order is a temporal proxy; duplicate rows are retained.")
    steps = [
        ("1 · Selection", "bank-additional-full: 41,188 rows and 21 columns; verified checksum and official data dictionary.", "audit.json"),
        ("2 · Preprocessing", "Validate categories and ranges; unknown is valid, blanks are rejected. Fit encoders and scalers only on training folds.", "split_manifest.json"),
        ("3 · Transformation", "10 core features. pdays=999 becomes previously_contacted=0 and pdays_value=0; pdays=0 keeps flag=1. duration, campaign, y, and row_id are excluded as features.", "feature_manifest.json"),
        ("4 · Data mining", "Five models, 41 candidates, and 3 expanding-window folds; inner-CV AP. Fit sigmoid calibration using FrozenEstimator on calibration data only.", "cv_results.csv"),
        ("5 · Interpretation", "Selection chooses the winner and thresholds; lock before a single final test. Top-K capacity differs from threshold classification.", "selection_lock.json"),
    ]
    for title, text, filename in steps:
        with st.expander(title, expanded=True):
            st.write(text)
            if run:
                _download(run / filename, f"Artifact {filename}")
    if not run:
        st.info("Audit and splits unavailable.")
        return
    audit = read_json(run / "audit.json")
    counts = audit["counts"]
    st.write(f"Duplicates: {counts['duplicate_rows']} · errors: {counts['errors']} · warnings: {counts['warnings']}. Inconsistent values are flagged, never silently corrected.")
    st.dataframe(pd.DataFrame([{"column": key, "unknown": value} for key, value in counts["unknown"].items()]), hide_index=True)
    st.json(audit.get("summary", audit.get("descriptive", {})))
    split = read_json(run / "split_manifest.json")
    st.subheader("Half-open splits without shuffling")
    st.dataframe(pd.DataFrame([{"split": key, "start": value[0], "end_exclusive": value[1], **split["counts"][key]} for key, value in split["ranges"].items()]), hide_index=True)
    st.write(split["warning"])
    st.json(split["duplicate_overlap"])


def comparison(root, run, ref):
    st.title("Model Comparison")
    if not run:
        st.info("Comparison artifacts unavailable.")
        return
    st.caption(f"Feature set {ref.feature_set}. Pre-test winner remains {ref.model_id}; sorting does not change the active model.")
    scope = st.selectbox("Metric source", ["Inner-CV train", "Validation-selection", "Final test"])
    filename = {"Inner-CV train": "cv_results.csv", "Validation-selection": "selection_metrics.csv", "Final test": "test_metrics.csv"}[scope]
    if not (run / filename).exists():
        st.info("N/A: stage not yet run.")
        return
    table = pd.read_csv(run / filename)
    metric = st.selectbox("Sort by metric", [name for name in ("average_precision", "precision_at_k", "brier_score", "mean_ap", "f1", "roc_auc") if name in table])
    st.dataframe(table.sort_values(metric, ascending=metric == "brier_score", kind="stable"), hide_index=True)
    chart = table[table.selected] if "selected" in table else table
    _plot(chart, "model_id", metric, scope, "bar")
    st.subheader("Fit runtime / parameters and calibration")
    metadata = []
    lock = read_json(run / "selection_lock.json")
    for model_id in lock["candidate_model_hashes"]:
        m = read_json(run / "models" / model_id / "model_metadata.json")
        metadata.append({"model_id": model_id, "fit_seconds": m["fit_seconds"], "calibration_seconds": m["calibration_seconds"], "threshold": m["threshold"], "inner_cv_ap": m["best_inner_cv_ap"], "hyperparameters": str(m["hyperparameters"])})
    st.dataframe(pd.DataFrame(metadata), hide_index=True)
    _download(run / filename)


def single(root, run, ref):
    st.title("Single Prediction")
    st.write("Use a synthetic example or authorized record with 10 core features. unknown is accepted; invalid values are not replaced with unknown.")
    record = {}
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            record["age"] = st.number_input("age · years", min_value=1, max_value=120, value=40, step=1)
            for name in ("job", "marital", "education", "default"):
                record[name] = st.selectbox(name, CATEGORIES[name])
        with right:
            for name in ("housing", "loan", "poutcome"):
                record[name] = st.selectbox(name, CATEGORIES[name])
            record["pdays"] = st.number_input("pdays · 999 means never previously contacted", min_value=0, max_value=999, value=999, step=1)
            record["previous"] = st.number_input("previous · contacts before this campaign", min_value=0, value=0, step=1)
        submitted = st.button("Predict", disabled=ref is None, type="primary")
    key = hashlib.sha256(canonical_bytes({"input": record, "model_hash": ref.model_hash if ref else None,
                                        "run_id": ref.run_id if ref else None, "threshold": ref.threshold if ref else None})).hexdigest()
    if st.session_state.get("single_key") != key:
        st.session_state.pop("single_result", None)
    if submitted:
        with st.spinner("Scoring one observation..."):
            st.session_state.single_result = score_records(pd.DataFrame([record]), ref)
            st.session_state.single_key = key
    result = st.session_state.get("single_result")
    if result:
        row = result["rows"][0]
        a, b = st.columns(2)
        a.metric("Estimated p_yes", f"{row['p_yes']:.2%}")
        b.metric("Label threshold", row["predicted_label"])
        st.caption(f"{result['run_id']} / {result['model_id']} / {result['feature_set']} / threshold {result['threshold']:.2f}")
        if result["warnings"]:
            st.dataframe(pd.DataFrame(result["warnings"]), hide_index=True)
        st.download_button("Download result JSON", export_results(result, "json"), "single_score.json", "application/json")
    if ref is None:
        st.info("model-not-ready: prediction disabled; complete offline setup.")


def batch(root, run, ref):
    st.title("Batch Priorities")
    st.caption("UTF-8 CSV, delimiter ; or , · maximum 20 MB / 50,000 rows · all-or-nothing")
    input_epoch = st.session_state.get("input_epoch", 0)
    demo = st.checkbox("Labeled y demo mode · not production evaluation", value=False, key=f"demo_{input_epoch}")
    upload = st.file_uploader("Upload core CSV", type=["csv"], key=f"upload_{input_epoch}")
    sample = st.selectbox("Or a local example", ["No example", "demo_valid.csv", "demo_invalid.csv", "Labeled public subset"], key=f"sample_{input_epoch}")
    if sample == "Labeled public subset" and upload is None:
        st.caption("Source: public UCI raw data, 100 calibration observations starting at row_id=24712. Demo example, not a new holdout.")
        if not demo:
            st.warning("Enable labeled demo mode for this subset.")
        raw_path = root / "data/raw/bank-additional-full.csv"
        content = pd.read_csv(raw_path, sep=";").iloc[24712:24812].to_csv(index=False, sep=";").encode("utf-8") if raw_path.exists() else None
    else:
        content = upload.getvalue() if upload else (root / "tests" / "fixtures" / sample).read_bytes() if sample != "No example" else None
    for name in ("demo_valid.csv", "demo_invalid.csv"):
        _download(root / "tests" / "fixtures" / name, f"Download synthetic template {name}")
    if content is None:
        st.session_state.pop("batch_scores", None)
        st.session_state.pop("batch_key", None)
        st.info("empty: upload CSV or select a local example.")
        return
    identity = {"demo": demo, "run_id": ref.run_id if ref else None, "model_id": ref.model_id if ref else None,
                "model_hash": ref.model_hash if ref else None, "threshold": ref.threshold if ref else None}
    key = hashlib.sha256(content + canonical_bytes(identity)).hexdigest()
    if key != st.session_state.get("batch_key"):
        st.session_state.pop("batch_scores", None)
        st.session_state.batch_key = key
    with st.spinner("validating: checking the entire batch..."):
        records = read_csv(content)
        report = audit_dataset(records, "demo" if demo else "scoring")
    if not report["valid"]:
        st.error("invalid: entire batch rejected. Correct the rows/fields below; no partial predictions.")
        st.dataframe(pd.DataFrame(report["errors"]), hide_index=True)
        st.download_button("Download error CSV", export_errors(report), "validation_errors.csv", "text/csv")
        return
    st.success(f"ready: {len(records):,} valid rows · {len(report['warnings'])} warning")
    if report["warnings"]:
        with st.expander("Validation warnings / demo columns explicitly removed"):
            st.dataframe(pd.DataFrame(report["warnings"]), hide_index=True)
    if ref is None:
        st.info("model-not-ready: scoring unavailable.")
        return
    st.caption(f"{ref.run_id} / {ref.model_id} / {ref.feature_set} / threshold {ref.threshold:.2f} · reject_batch")
    mode = st.radio("Capacity mode", ["Fraction q", "Count K"], horizontal=True)
    if mode == "Fraction q":
        capacity = {"q": st.slider("Contact fraction q", 0.0, 1.0, 0.10, 0.01)}
    else:
        capacity = {"k": st.number_input("Priority count K", min_value=0, max_value=len(records), value=min(1000, len(records)), step=1)}
    if st.button("Score batch", type="primary"):
        with st.spinner("scoring: local model without training..."):
            st.session_state.batch_scores = score_records(records, ref, "demo" if demo else "scoring")
    scores = st.session_state.get("batch_scores")
    if not scores:
        return
    result = rank_records(scores, capacity)
    st.success(f"complete: exactly {result['capacity_k']:,} priorities out of {len(records):,} observations; actual q {result['call_fraction']:.2%}")
    st.caption("Changing K/q only updates ranking; no retraining. A yes label is not the same as selected_for_contact.")
    table = pd.DataFrame(result["rows"]).drop(columns="warnings")
    table["p_yes"] = table["p_yes"].map(lambda value: f"{value:.2%}")
    st.dataframe(table, hide_index=True)
    top_only = st.checkbox("Export top-K only", False)
    features = records.loc[:, list(CORE_FEATURES)] if st.checkbox("Include original features in export", False) else None
    st.download_button("Download priorities.csv", export_results(result, "csv", top_only, features), "priorities.csv", "text/csv")
    st.download_button("Download priorities.json", export_results(result, "json", top_only, features), "priorities.json", "application/json")
    st.download_button("Download batch_metadata.json", export_metadata(result), "batch_metadata.json", "application/json")
    if demo:
        st.warning("LABELED DEMO: evaluation of this upload, not the experiment's final test or real bank performance.")
        labels = records.y.str.strip().map({"no": 0, "yes": 1}).to_numpy()
        ordered = sorted(scores["rows"], key=lambda row: row["input_row_id"])
        evaluation = evaluate_scores(labels, [row["p_yes"] for row in ordered], ref.threshold, k=result["capacity_k"])
        st.json(evaluation["metrics"])
    else:
        st.info("Upload performance metrics N/A: no y labels. Validation metrics are not results for this batch.")


def evaluation(root, run, ref):
    st.title("Evaluation and Interpretation")
    if not run:
        st.info("N/A: no locked run.")
        return
    lock = read_json(run / "selection_lock.json")
    scope = st.selectbox("Evaluation split", ["selection", "test"])
    model = st.selectbox("Evaluation model", list(lock["candidate_model_hashes"]) + ["dummy_prior", "random_ranking"])
    path = run / "evaluations" / f"{scope}_{model}.json"
    if not path.exists():
        st.info("N/A: not evaluated on this split.")
        return
    report = read_json(path)
    st.caption(f"Scope {scope} · winner remains {lock['winner_model_id']} · threshold chosen before test · not new upload performance")
    metrics = report["metrics"]
    a, b, c, d = st.columns(4)
    a.metric("Average Precision (AP)", "N/A" if metrics["average_precision"] is None else f"{metrics['average_precision']:.4f}")
    b.metric("ROC-AUC", "N/A" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.4f}")
    c.metric("Brier score", "N/A" if metrics["brier_score"] is None else f"{metrics['brier_score']:.4f}")
    d.metric("K / N", f"{metrics['capacity_k']} / {metrics['n']}")
    st.caption("Average Precision (AP) summarizes precision-recall, not trapezoidal area. Lower Brier is better. Threshold and capacity are separate.")
    st.json(metrics)
    if report.get("note"):
        st.info("Random ranking with seed=42; scores are not probabilities. Probability and classification metrics are N/A." if model == "random_ranking" else report["note"])
    if report.get("metrics_at_0_5"):
        with st.expander("Additional threshold 0.5 comparison / not the active threshold"):
            st.json(report["metrics_at_0_5"])
    st.write("Confusion matrix: actual rows, predicted columns [no, yes]")
    if report.get("confusion_matrix") is not None:
        st.dataframe(pd.DataFrame(report["confusion_matrix"], index=["actual no", "actual yes"], columns=["predicted no", "predicted yes"]))
    else:
        st.info("N/A: random ranking baseline has no classification threshold.")
    for name in ("pr", "roc", "calibration", "capacity", "confusion", "score_distribution"):
        chart = run / "charts" / f"{scope}_{model}_{name}.png"
        if chart.exists():
            st.image(str(chart), caption=f"{scope} / {model} / {name}")
            _download(chart)
        elif name != "confusion":
            curves = report["curves"]
            if name == "pr":
                values = curves["precision_recall"]
                _plot(pd.DataFrame({"recall": values["recall"], "precision": values["precision"]}), "recall", "precision", "Precision-recall")
            elif name == "roc":
                values = curves["roc"]
                _plot(pd.DataFrame({"fpr": values["fpr"], "tpr": values["tpr"]}), "fpr", "tpr", "ROC")
            elif name == "calibration":
                _plot(pd.DataFrame(curves["calibration"]), "mean_predicted", "observed", "Calibration / 10 quantile bins")
            elif name == "capacity":
                values = pd.DataFrame(curves["capacity"])
                _plot(values.melt(id_vars="fraction", value_vars=["precision_at_k", "recall_at_k", "lift_at_k"], var_name="metric", value_name="value"), "fraction", "value", "Ranking quality / capacity", color="metric")
    st.subheader("Calibration bin counts and explicit capacity")
    st.dataframe(pd.DataFrame(report["curves"]["calibration"]), hide_index=True)
    st.dataframe(pd.DataFrame(report["curves"]["capacity"]), hide_index=True)
    _download(path, "Download metrics and curves JSON")
    if (run / "interpretation.json").exists():
        with st.expander("Permutation importance and subgroups on selection"):
            info = read_json(run / "interpretation.json")
            st.dataframe(pd.DataFrame(info["permutation_importance"]).drop(columns="repeats"), hide_index=True)
            st.dataframe(pd.DataFrame(info["subgroups"]).drop(columns=["metrics", "calibration"]), hide_index=True)
            st.write("Importance includes negative values; correlated features limit interpretation. These are not causal effects.")
            st.write("Subgroups with n<100 have insufficient evidence. Limited metrics do not establish fairness.")
            st.write("Only selection data is used; the locked model, thresholds, and test baseline remain unchanged.")
            _download(run / "interpretation.json")
    st.warning("Prevalence drift, correlated features, and small groups limit interpretation. Fairness is not established; do not use for credit decisions. The dataset contains neither realized ROI nor consent records.")


def documentation(root, run, ref):
    st.title("Documentation")
    st.write("COMP6140001 Data Mining · BINUS University · local academic prototype. Requirements describe the implemented baseline, not instructor approval.")
    st.markdown("[UCI dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) · [Moro, Cortez & Rita (2014)](https://doi.org/10.1016/j.dss.2014.03.001) · license CC BY 4.0")
    st.subheader("Offline setup and demo")
    st.code("py -3.11 -m venv .venv\n.venv/Scripts/python.exe -m pip install -r requirements.lock\n.venv/Scripts/python.exe -m pip install -e . --no-deps\n.venv/Scripts/python.exe scripts/restore_demo.py\n.venv/Scripts/python.exe -m streamlit run app.py")
    st.write("Runner stages: ingest, audit, split, train, calibrate, select, freeze, test, activate. Evaluate the baseline test once; reproduce verifies saved predictions without changing decisions.")
    st.write("Demo: Data and KDD; selection/test comparison; single form; valid CSV at q=10%; invalid CSV; exports and manifests. No joblib/pickle uploads. Batches remain in session memory only.")
    if st.button("Clear session data"):
        for key in ("single_key", "single_result", "batch_key", "batch_scores"):
            st.session_state.pop(key, None)
        st.session_state.input_epoch = st.session_state.get("input_epoch", 0) + 1
        st.success("Uploaded data and session results cleared; public raw data and models remain intact.")
    if run:
        st.subheader("Provenance / manifest")
        st.json(read_json(run / "run_manifest.json"))
        for filename in ("run_manifest.json", "selection_lock.json", "split_manifest.json", "config_resolved.yaml", "environment.json", "benchmark.json", "change_log.json"):
            _download(run / filename)
    st.subheader("Academic submission files")
    for path in (root / "reports/final_report.pdf", root / "presentation/slides.pptx", root / "presentation/slides.pdf", root / "reports/experiment_summary.json"):
        _download(path)
    st.warning("Before submission: add member names/NIM, publish and check public links without login, and complete the presentation, video, and proposal separately. See docs/SUBMISSION_CHECKLIST.md.")
    st.write("context_v1 is an unrun retrospective experiment; the main winner remains core_v1. Changes after test require a new version and holdout.")


def render_app(root):
    st.markdown("""<style>
    h1,h2,h3 {font-family:Georgia,serif;letter-spacing:-.025em;color:#172D43}
    h1 {font-size:2.8rem!important;max-width:900px;line-height:1.1!important}
    [data-testid=stSidebar] {border-right:1px solid #d7e1e7}
    [data-testid=stMetric] {border-top:3px solid #087F83;padding:16px 6px;background:#eef3f5}
    [data-testid=stMetricLabel] {font-size:14px} .stMainBlockContainer {padding-top:2.5rem}
    button {min-height:42px} p,li,label {font-size:16px}
    </style>""", unsafe_allow_html=True)
    st.sidebar.markdown("## DepoRank\nCAMPAIGN ANALYTICS\n\n---")
    page = st.sidebar.radio("Navigation", PAGES)
    ref, run = None, None
    try:
        ref = Registry(root / "artifacts").active()
        run = Registry(root / "artifacts").run_path(ref.run_id)
        st.sidebar.success(f"Model ready\n\n{ref.model_id}\n\n{ref.feature_set}")
    except DomainError as error:
        st.session_state.pop("batch_scores", None)
        st.session_state.pop("single_result", None)
        show_error(error)
    except Exception:
        error = DomainError("INTERNAL_ERROR", "Startup failed")
        LOG.error("state=startup_error error_id=%s", error.error_id)
        show_error(error)
    st.sidebar.caption("Local · no UI training\n\nRanks observations, not unique customers")
    renderers = (overview, data_kdd, comparison, single, batch, evaluation, documentation)
    try:
        renderers[PAGES.index(page)](root, run, ref)
    except DomainError as error:
        st.session_state.pop("batch_scores", None)
        show_error(error)
    except Exception:
        error = DomainError("INTERNAL_ERROR", "Page failed")
        LOG.error("state=page_error error_id=%s page=%s", error.error_id, page)
        show_error(error)

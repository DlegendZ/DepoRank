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

PAGES = ("Ringkasan", "Data dan KDD", "Perbandingan Model", "Prediksi Tunggal", "Prioritas Batch", "Evaluasi dan Interpretasi", "Dokumentasi")
LOG = logging.getLogger(__name__)
MESSAGES = {
    "MODEL_NOT_READY": "Model belum siap. Jalankan runner offline sampai freeze dan activate; dokumentasi tetap tersedia.",
    "ARTIFACT_INTEGRITY_ERROR": "Integritas artefak gagal. Prediksi dihentikan; periksa registry melalui CLI verify.",
    "ENVIRONMENT_MISMATCH": "Environment berbeda. Gunakan Python dan requirements.lock yang sesuai run.",
    "SCHEMA_INVALID": "Schema tidak sesuai. Periksa header, format CSV, kolom wajib dan mode input.",
    "ROW_INVALID": "Baris tidak valid. Seluruh batch ditolak; perbaiki kolom pada laporan error.",
    "CAPACITY_INVALID": "Kapasitas tidak valid. Gunakan satu nilai K atau q dalam batas batch.",
}


def show_error(error):
    st.error(f"{error.code}: {MESSAGES.get(error.code, 'Operasi gagal. Hubungi pengelola dengan error ID.')} ID: {error.error_id}")
    LOG.warning("state=error code=%s error_id=%s", error.code, error.error_id)
    if isinstance(error.details, dict) and error.details.get("errors"):
        st.dataframe(pd.DataFrame(error.details["errors"]), hide_index=True)
        st.download_button("Unduh laporan error", export_errors(error.details), "validation_errors.csv", "text/csv")


def _plot(frame, x, y, title, kind="line", color=None):
    if frame.empty:
        st.info("N/A: data grafik belum tersedia.")
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
    st.title("Prioritas tepat. Kapasitas terbatas.")
    st.write("DepoRank memeringkat observasi kontak kampanye deposito. Analisis lokal, keputusan model dapat diaudit, tanpa training saat demo.")
    if run is None:
        st.info("Empty: belum ada run terkunci. Buka Dokumentasi untuk setup.")
        return
    manifest = read_json(run / "run_manifest.json")
    split = read_json(run / "split_manifest.json")
    audit = read_json(run / "audit.json")
    selection = pd.read_csv(run / "selection_metrics.csv")
    winner = selection[selection.model_id == ref.model_id].iloc[0]
    columns = st.columns(4)
    columns[0].metric("Observasi publik", f"{audit['counts']['rows']:,}")
    columns[1].metric("Target yes / seluruh data", f"{audit['counts']['target']['yes'] / audit['counts']['rows']:.2%}")
    columns[2].metric("Model dipilih pada selection", ref.model_id)
    columns[3].metric("AP selection", f"{winner.average_precision:.4f}")
    st.success(f"{manifest['status']} · {ref.run_id} · {ref.feature_set} · threshold {ref.threshold:.2f} · 5 keluarga model")
    st.subheader("Protokol empat blok, satu keputusan terkunci")
    frame = pd.DataFrame([{ "split": name, **values } for name, values in split["counts"].items()])
    _plot(frame, "split", "prevalence", "Prevalence berubah antar blok", "bar")
    st.dataframe(frame, hide_index=True)
    if (run / "test_metrics.csv").exists():
        st.subheader("Final test / bukan pemilihan ulang")
        test = pd.read_csv(run / "test_metrics.csv")
        selected = test[test.model_id == ref.model_id].iloc[0]
        a, b, c = st.columns(3)
        a.metric("AP test", f"{selected.average_precision:.4f}")
        b.metric(f"Precision@{int(selected.capacity_k)}", f"{selected.precision_at_k:.2%}")
        c.metric(f"Lift@{int(selected.capacity_k)}", f"{selected.lift_at_k:.2f}")
    st.warning("Portugal 2008–2010; bukan kesiapan produksi Indonesia. Probabilitas bukan kepastian, ranking bukan uplift kausal, observasi bukan nasabah unik.")
    st.markdown("Sumber: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing). Lisensi CC BY 4.0.")


def data_kdd(root, run, ref):
    st.title("Data dan KDD")
    st.write("Raw publik read-only. Urutan asli dipakai sebagai proksi waktu; duplikasi dipertahankan.")
    steps = [
        ("1 · Selection", "Varian bank-additional-full: 41.188 baris, 21 kolom; checksum dan kamus resmi.", "audit.json"),
        ("2 · Preprocessing", "Validasi kategori/rentang; unknown sah, blank ditolak. Encoder/scaler fit hanya fold train.", "split_manifest.json"),
        ("3 · Transformation", "10 fitur core. pdays=999 menjadi flag previously_contacted=0 dan pdays_value=0; pdays=0 memiliki flag=1. duration/campaign/y/row_id dilarang sebagai fitur.", "feature_manifest.json"),
        ("4 · Data mining", "Lima model, 41 kandidat, 3 expanding-window fold; AP inner-CV. Sigmoid FrozenEstimator pada calibration saja.", "cv_results.csv"),
        ("5 · Interpretation", "Selection memilih winner dan threshold; lock sebelum test sekali. Kapasitas top-K berbeda dari label threshold.", "selection_lock.json"),
    ]
    for title, text, filename in steps:
        with st.expander(title, expanded=True):
            st.write(text)
            if run:
                _download(run / filename, f"Artefak {filename}")
    if not run:
        st.info("Audit dan split belum tersedia.")
        return
    audit = read_json(run / "audit.json")
    counts = audit["counts"]
    st.write(f"Duplikasi: {counts['duplicate_rows']} · errors: {counts['errors']} · warnings: {counts['warnings']}. Nilai inconsistent diberi warning, tidak dibetulkan diam-diam.")
    st.dataframe(pd.DataFrame([{"kolom": key, "unknown": value} for key, value in counts["unknown"].items()]), hide_index=True)
    st.json(audit.get("summary", audit.get("descriptive", {})))
    split = read_json(run / "split_manifest.json")
    st.subheader("Split half-open, tanpa shuffle")
    st.dataframe(pd.DataFrame([{"split": key, "start": value[0], "end_exclusive": value[1], **split["counts"][key]} for key, value in split["ranges"].items()]), hide_index=True)
    st.write(split["warning"])
    st.json(split["duplicate_overlap"])


def comparison(root, run, ref):
    st.title("Perbandingan Model")
    if not run:
        st.info("Belum ada artefak comparison.")
        return
    st.caption(f"Feature set {ref.feature_set}. Winner pra-test tetap {ref.model_id}; sorting tidak mengganti model aktif.")
    scope = st.selectbox("Sumber metrik", ["Inner-CV train", "Validation-selection", "Final test"])
    filename = {"Inner-CV train": "cv_results.csv", "Validation-selection": "selection_metrics.csv", "Final test": "test_metrics.csv"}[scope]
    if not (run / filename).exists():
        st.info("N/A: tahap belum dijalankan.")
        return
    table = pd.read_csv(run / filename)
    metric = st.selectbox("Urutkan metrik", [name for name in ("average_precision", "precision_at_k", "brier_score", "mean_ap", "f1", "roc_auc") if name in table])
    st.dataframe(table.sort_values(metric, ascending=metric == "brier_score", kind="stable"), hide_index=True)
    chart = table[table.selected] if "selected" in table else table
    _plot(chart, "model_id", metric, scope, "bar")
    st.subheader("Fit runtime / parameter dan calibration")
    metadata = []
    lock = read_json(run / "selection_lock.json")
    for model_id in lock["candidate_model_hashes"]:
        m = read_json(run / "models" / model_id / "model_metadata.json")
        metadata.append({"model_id": model_id, "fit_seconds": m["fit_seconds"], "calibration_seconds": m["calibration_seconds"], "threshold": m["threshold"], "inner_cv_ap": m["best_inner_cv_ap"], "hyperparameters": str(m["hyperparameters"])})
    st.dataframe(pd.DataFrame(metadata), hide_index=True)
    _download(run / filename)


def single(root, run, ref):
    st.title("Prediksi Tunggal")
    st.write("Contoh sintetis atau catatan sah berisi 10 fitur core. unknown tersedia; kesalahan tidak diganti menjadi unknown.")
    record = {}
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            record["age"] = st.number_input("age · umur", min_value=1, max_value=120, value=40, step=1)
            for name in ("job", "marital", "education", "default"):
                record[name] = st.selectbox(name, CATEGORIES[name])
        with right:
            for name in ("housing", "loan", "poutcome"):
                record[name] = st.selectbox(name, CATEGORIES[name])
            record["pdays"] = st.number_input("pdays · 999 berarti belum pernah dihubungi", min_value=0, max_value=999, value=999, step=1)
            record["previous"] = st.number_input("previous · kontak sebelum kampanye ini", min_value=0, value=0, step=1)
        submitted = st.button("Prediksi", disabled=ref is None, type="primary")
    key = hashlib.sha256(canonical_bytes({"input": record, "model_hash": ref.model_hash if ref else None,
                                        "run_id": ref.run_id if ref else None, "threshold": ref.threshold if ref else None})).hexdigest()
    if st.session_state.get("single_key") != key:
        st.session_state.pop("single_result", None)
    if submitted:
        with st.spinner("Scoring satu observasi…"):
            st.session_state.single_result = score_records(pd.DataFrame([record]), ref)
            st.session_state.single_key = key
    result = st.session_state.get("single_result")
    if result:
        row = result["rows"][0]
        a, b = st.columns(2)
        a.metric("Estimasi p_yes", f"{row['p_yes']:.2%}")
        b.metric("Label threshold", row["predicted_label"])
        st.caption(f"{result['run_id']} / {result['model_id']} / {result['feature_set']} / threshold {result['threshold']:.2f}")
        if result["warnings"]:
            st.dataframe(pd.DataFrame(result["warnings"]), hide_index=True)
        st.download_button("Unduh hasil JSON", export_results(result, "json"), "single_score.json", "application/json")
    if ref is None:
        st.info("model-not-ready: prediksi disabled; lakukan setup offline.")


def batch(root, run, ref):
    st.title("Prioritas Batch")
    st.caption("CSV UTF-8, delimiter ; atau , · maksimal 20 MB / 50.000 baris · all-or-nothing")
    input_epoch = st.session_state.get("input_epoch", 0)
    demo = st.checkbox("Mode demo berlabel y · bukan evaluasi produksi", value=False, key=f"demo_{input_epoch}")
    upload = st.file_uploader("Unggah CSV core", type=["csv"], key=f"upload_{input_epoch}")
    sample = st.selectbox("Atau contoh lokal", ["Tidak memakai contoh", "demo_valid.csv", "demo_invalid.csv", "Subset publik berlabel"], key=f"sample_{input_epoch}")
    if sample == "Subset publik berlabel" and upload is None:
        st.caption("Sumber: UCI raw publik, 100 observasi calibration mulai row_id=24712. Contoh demonstrasi, bukan holdout baru.")
        if not demo:
            st.warning("Aktifkan mode demo berlabel untuk subset ini.")
        raw_path = root / "data/raw/bank-additional-full.csv"
        content = pd.read_csv(raw_path, sep=";").iloc[24712:24812].to_csv(index=False, sep=";").encode("utf-8") if raw_path.exists() else None
    else:
        content = upload.getvalue() if upload else (root / "tests" / "fixtures" / sample).read_bytes() if sample != "Tidak memakai contoh" else None
    for name in ("demo_valid.csv", "demo_invalid.csv"):
        _download(root / "tests" / "fixtures" / name, f"Unduh template sintetis {name}")
    if content is None:
        st.session_state.pop("batch_scores", None)
        st.session_state.pop("batch_key", None)
        st.info("empty: unggah CSV atau pilih contoh lokal.")
        return
    identity = {"demo": demo, "run_id": ref.run_id if ref else None, "model_id": ref.model_id if ref else None,
                "model_hash": ref.model_hash if ref else None, "threshold": ref.threshold if ref else None}
    key = hashlib.sha256(content + canonical_bytes(identity)).hexdigest()
    if key != st.session_state.get("batch_key"):
        st.session_state.pop("batch_scores", None)
        st.session_state.batch_key = key
    with st.spinner("validating: periksa seluruh batch…"):
        records = read_csv(content)
        report = audit_dataset(records, "demo" if demo else "scoring")
    if not report["valid"]:
        st.error("invalid: seluruh batch ditolak. Perbaiki baris/kolom berikut; tidak ada prediksi parsial.")
        st.dataframe(pd.DataFrame(report["errors"]), hide_index=True)
        st.download_button("Unduh error CSV", export_errors(report), "validation_errors.csv", "text/csv")
        return
    st.success(f"ready: {len(records):,} baris valid · {len(report['warnings'])} warning")
    if report["warnings"]:
        with st.expander("Warning validasi / kolom demo dibuang secara eksplisit"):
            st.dataframe(pd.DataFrame(report["warnings"]), hide_index=True)
    if ref is None:
        st.info("model-not-ready: tidak dapat scoring.")
        return
    st.caption(f"{ref.run_id} / {ref.model_id} / {ref.feature_set} / threshold {ref.threshold:.2f} · reject_batch")
    mode = st.radio("Mode kapasitas", ["Proporsi q", "Jumlah K"], horizontal=True)
    if mode == "Proporsi q":
        capacity = {"q": st.slider("Proporsi kontak q", 0.0, 1.0, 0.10, 0.01)}
    else:
        capacity = {"k": st.number_input("Jumlah prioritas K", min_value=0, max_value=len(records), value=min(1000, len(records)), step=1)}
    if st.button("Score batch", type="primary"):
        with st.spinner("scoring: model lokal tanpa training…"):
            st.session_state.batch_scores = score_records(records, ref, "demo" if demo else "scoring")
    scores = st.session_state.get("batch_scores")
    if not scores:
        return
    result = rank_records(scores, capacity)
    st.success(f"complete: tepat {result['capacity_k']:,} prioritas dari {len(records):,} observasi; q aktual {result['call_fraction']:.2%}")
    st.caption("K/q berubah hanya ranking; bukan retraining. Label yes tidak sama dengan selected_for_contact.")
    table = pd.DataFrame(result["rows"]).drop(columns="warnings")
    table["p_yes"] = table["p_yes"].map(lambda value: f"{value:.2%}")
    st.dataframe(table, hide_index=True)
    top_only = st.checkbox("Ekspor hanya top-K", False)
    features = records.loc[:, list(CORE_FEATURES)] if st.checkbox("Sertakan fitur asli dalam ekspor", False) else None
    st.download_button("Unduh priorities.csv", export_results(result, "csv", top_only, features), "priorities.csv", "text/csv")
    st.download_button("Unduh priorities.json", export_results(result, "json", top_only, features), "priorities.json", "application/json")
    st.download_button("Unduh batch_metadata.json", export_metadata(result), "batch_metadata.json", "application/json")
    if demo:
        st.warning("DEMO berlabel: evaluasi upload ini, bukan final test eksperimen atau performa bank nyata.")
        labels = records.y.str.strip().map({"no": 0, "yes": 1}).to_numpy()
        ordered = sorted(scores["rows"], key=lambda row: row["input_row_id"])
        evaluation = evaluate_scores(labels, [row["p_yes"] for row in ordered], ref.threshold, k=result["capacity_k"])
        st.json(evaluation["metrics"])
    else:
        st.info("Metrik performa upload N/A: tidak ada label y. Metrik validation tidak dianggap hasil batch ini.")


def evaluation(root, run, ref):
    st.title("Evaluasi dan Interpretasi")
    if not run:
        st.info("N/A: belum ada run terkunci.")
        return
    lock = read_json(run / "selection_lock.json")
    scope = st.selectbox("Split evaluasi", ["selection", "test"])
    model = st.selectbox("Model evaluasi", list(lock["candidate_model_hashes"]) + ["dummy_prior", "random_ranking"])
    path = run / "evaluations" / f"{scope}_{model}.json"
    if not path.exists():
        st.info("N/A: belum dievaluasi pada split tersebut.")
        return
    report = read_json(path)
    st.caption(f"Scope {scope} · winner tetap {lock['winner_model_id']} · threshold dipilih pra-test · bukan performa upload baru")
    metrics = report["metrics"]
    a, b, c, d = st.columns(4)
    a.metric("Average Precision (AP)", "N/A" if metrics["average_precision"] is None else f"{metrics['average_precision']:.4f}")
    b.metric("ROC-AUC", "N/A" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.4f}")
    c.metric("Brier score", "N/A" if metrics["brier_score"] is None else f"{metrics['brier_score']:.4f}")
    d.metric("K / N", f"{metrics['capacity_k']} / {metrics['n']}")
    st.caption("Average Precision (AP) merangkum precision-recall, bukan luas trapezoidal. Brier lebih kecil lebih baik. Threshold dan kapasitas berbeda.")
    st.json(metrics)
    if report.get("note"):
        st.info(report["note"])
    if report.get("metrics_at_0_5"):
        with st.expander("Pembanding tambahan threshold 0.5 / bukan threshold aktif"):
            st.json(report["metrics_at_0_5"])
    st.write("Confusion matrix: baris aktual, kolom prediksi [no, yes]")
    if report.get("confusion_matrix") is not None:
        st.dataframe(pd.DataFrame(report["confusion_matrix"], index=["aktual no", "aktual yes"], columns=["prediksi no", "prediksi yes"]))
    else:
        st.info("N/A: baseline ranking acak tidak memiliki threshold klasifikasi.")
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
                _plot(values.melt(id_vars="fraction", value_vars=["precision_at_k", "recall_at_k", "lift_at_k"], var_name="metric", value_name="value"), "fraction", "value", "Kualitas ranking / kapasitas", color="metric")
    st.subheader("Calibration bin counts dan kapasitas eksplisit")
    st.dataframe(pd.DataFrame(report["curves"]["calibration"]), hide_index=True)
    st.dataframe(pd.DataFrame(report["curves"]["capacity"]), hide_index=True)
    _download(path, "Unduh metrics dan curves JSON")
    if (run / "interpretation.json").exists():
        with st.expander("P1 · importance dan subgroup pada selection"):
            info = read_json(run / "interpretation.json")
            st.dataframe(pd.DataFrame(info["permutation_importance"]).drop(columns="repeats"), hide_index=True)
            st.dataframe(pd.DataFrame(info["subgroups"]).drop(columns=["metrics", "calibration"]), hide_index=True)
            for note in info["notes"]:
                st.write(note)
            _download(run / "interpretation.json")
    st.warning("Drift prevalence, feature correlation dan kelompok kecil membatasi interpretasi. Fairness tidak dijamin; jangan gunakan sebagai keputusan kredit. Tidak ada ROI aktual atau consent dalam dataset.")


def documentation(root, run, ref):
    st.title("Dokumentasi")
    st.write("COMP6140001 Data Mining · BINUS · prototipe lokal. SRD v1.0 merupakan baseline usulan, bukan persetujuan dosen.")
    st.markdown("[UCI dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) · [Moro, Cortez & Rita (2014)](https://doi.org/10.1016/j.dss.2014.03.001) · lisensi CC BY 4.0")
    st.subheader("Setup offline dan demo")
    st.code("uv venv --python 3.11\nuv pip sync requirements.lock --python .venv/Scripts/python.exe\nuv pip install --python .venv/Scripts/python.exe -e . --no-deps\n.venv/Scripts/python.exe -m bank_marketing.cli --help\n.venv/Scripts/python.exe -m streamlit run app.py")
    st.write("Runner: ingest → audit → split → train → calibrate → select → freeze → test → activate. Test sekali untuk baseline; reproduce hanya memverifikasi prediksi tersimpan, tidak mengubah keputusan.")
    st.write("Demo: Data dan KDD; Perbandingan Model selection/test; form tunggal; CSV valid q=10%; CSV invalid; export dan manifest. Tidak ada upload joblib/pickle. Batch hanya disimpan dalam memori sesi.")
    if st.button("Hapus data sesi"):
        for key in ("single_key", "single_result", "batch_key", "batch_scores"):
            st.session_state.pop(key, None)
        st.session_state.input_epoch = st.session_state.get("input_epoch", 0) + 1
        st.success("Data upload dan hasil sesi dibuang; raw/model publik tetap utuh.")
    if run:
        st.subheader("Provenance / manifest")
        st.json(read_json(run / "run_manifest.json"))
        for filename in ("run_manifest.json", "selection_lock.json", "split_manifest.json", "config_resolved.yaml", "environment.json", "benchmark.json", "change_log.json"):
            _download(run / filename)
    st.subheader("Draft paket akademik")
    for path in (root / "reports/final_report.pdf", root / "reports/proposal.pdf", root / "presentation/slides.pdf", root / "reports/experiment_summary.json"):
        _download(path)
    st.warning("Pending manual: nama/NIM anggota, persetujuan dosen tentang NLP, durasi presentasi, tautan publik/incognito, rekaman video dan UAT keyboard/offline. Tidak ada publikasi otomatis.")
    st.write("context_v1 adalah eksperimen retrospektif terpisah yang belum dijalankan; winner utama tetap core_v1. Perubahan baseline setelah test perlu versi dan holdout baru.")


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
    page = st.sidebar.radio("Navigasi", PAGES)
    ref, run = None, None
    try:
        ref = Registry(root / "artifacts").active()
        run = Registry(root / "artifacts").run_path(ref.run_id)
        st.sidebar.success(f"Model siap\n\n{ref.model_id}\n\n{ref.feature_set}")
    except DomainError as error:
        st.session_state.pop("batch_scores", None)
        st.session_state.pop("single_result", None)
        show_error(error)
    except Exception:
        error = DomainError("INTERNAL_ERROR", "Startup failed")
        LOG.error("state=startup_error error_id=%s", error.error_id)
        show_error(error)
    st.sidebar.caption("Lokal · tanpa training UI\n\nRanking observasi, bukan nasabah unik")
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

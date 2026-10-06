from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

from bank_marketing.ml.registry import Registry
from bank_marketing.storage import read_json, utc_now, write_json

SOURCE = "https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing"
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip"


def render_charts(run_path):
    chart_dir = run_path / "charts"
    chart_dir.mkdir(exist_ok=True)
    output = []
    for item in sorted((run_path / "evaluations").glob("*.json")):
        report = read_json(item)
        curves = report["curves"]
        for kind in ("pr", "roc", "calibration", "capacity", "confusion"):
            if kind == "confusion" and report["confusion_matrix"] is None:
                continue
            if kind == "calibration" and not curves["calibration"]:
                continue
            fig, ax = plt.subplots(figsize=(6.5, 4))
            color = "#087e8b"
            if kind == "pr":
                values = curves["precision_recall"]
                ax.step(values["recall"], values["precision"], where="post", color=color)
                ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
            elif kind == "roc":
                values = curves["roc"]
                ax.plot(values["fpr"], values["tpr"], color=color)
                ax.plot([0, 1], [0, 1], "--", color="gray")
                ax.set(xlabel="False positive rate", ylabel="True positive rate")
            elif kind == "calibration":
                values = curves["calibration"]
                ax.plot([v["mean_predicted"] for v in values], [v["observed"] for v in values], "o-", color=color)
                ax.plot([0, 1], [0, 1], "--", color="gray")
                ax.set(xlabel="Rata-rata probabilitas", ylabel="Proporsi yes aktual")
            elif kind == "capacity":
                values = [v for v in curves["capacity"] if "policy" not in v]
                for metric in ("precision_at_k", "recall_at_k", "lift_at_k"):
                    ax.plot([v["fraction"] for v in values], [v[metric] for v in values], "o-", label=metric)
                ax.set(xlabel="Fraksi kapasitas", ylabel="Nilai metrik")
                ax.legend(fontsize=8)
            else:
                cm = report["confusion_matrix"]
                ax.imshow(cm, cmap="Blues")
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(cm[i][j]), ha="center", va="center")
                ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no", "yes"], yticklabels=["no", "yes"], xlabel="Prediksi", ylabel="Aktual")
            ax.set_title(f"{report['split']} / {report['model_id']} / {kind}", fontsize=11)
            fig.tight_layout()
            path = chart_dir / f"{report['split']}_{report['model_id']}_{kind}.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            output.append(str(path))
    return output


def generate_reports(run_id, root="artifacts"):
    registry = Registry(root)
    manifest = registry.verify_run(run_id)
    run = registry.run_path(run_id)
    lock = read_json(run / "selection_lock.json")
    selection = pd.read_csv(run / "selection_metrics.csv")
    test = pd.read_csv(run / "test_metrics.csv") if (run / "test_metrics.csv").exists() else None
    chart_paths = render_charts(run)
    reports = Path("reports")
    presentation = Path("presentation")
    reports.mkdir(exist_ok=True)
    presentation.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    styles["Title"].textColor = colors.HexColor("#123348")
    styles["Heading1"].textColor = colors.HexColor("#087e8b")
    story = []
    def paragraph(text, style="BodyText"):
        return Paragraph(escape(text).replace("\n", "<br/>"), styles[style])
    def add(text, style="BodyText"):
        story.extend([paragraph(text, style), Spacer(1, 10)])
    def table(frame):
        columns = [c for c in ("model_id", "average_precision", "precision_at_k", "recall_at_k", "lift_at_k", "brier_score", "f1") if c in frame]
        header = ["Model", "AP", "P@K", "R@K", "Lift@K", "Brier", "F1"][:len(columns)]
        rows = [header]
        for _, row in frame.iterrows():
            rows.append([paragraph(str(row[c]) if c == "model_id" else ("N/A" if pd.isna(row[c]) else f"{row[c]:.4f}")) for c in columns])
        result = Table(rows, repeatRows=1, colWidths=[145] + [55] * (len(columns) - 1))
        result.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123348")),
                                   ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        story.extend([result, Spacer(1, 14)])
    add("DepoRank", "Title")
    add("Sistem Analisis dan Prioritisasi Kampanye Deposito Berjangka", "Heading1")
    add("COMP6140001 Data Mining - BINUS University")
    add("DRAFT AKADEMIK - BELUM SIAP PENYERAHAN. Nama lengkap, NIM, kelas, persetujuan kasus/NLP, deadline, tautan publik dan video belum diberikan. Tidak mengarang identitas atau persetujuan.")
    add(f"Run: {run_id}\nFeature set: {lock['feature_set']}\nWinner pra-test: {lock['winner_model_id']}\nDibuat UTC: {utc_now()}")
    story.append(PageBreak())
    chapters = [
        ("Bab 1. Pendahuluan", "Kapasitas panggilan terbatas. Tujuan sistem adalah memprediksi asosiasi historis y=yes dan mengurutkan observasi menurut peluang berlangganan deposito. Unit analisis adalah observasi kontak, bukan nasabah unik. Prototipe akademik lokal, bukan keputusan kredit atau layanan bank produksi."),
        ("Bab 2. Landasan teori", "KDD meliputi selection, preprocessing, transformation, data mining, interpretation/evaluation. Data warehouse mengintegrasikan riwayat lintas sumber untuk analisis; direktori raw dan artefak lokal bukan enterprise warehouse. MVP satu sumber tidak membutuhkan warehouse. Logistic Regression linear; GaussianNB memakai asumsi independensi dan distribusi Gaussian; Decision Tree menghasilkan aturan; Random Forest menggabungkan pohon; Gradient Boosting membangun ensemble bertahap. AP adalah average_precision_score, bukan integral trapezoidal PR. Imbalance membuat accuracy saja tidak memadai. Precision/recall/lift@K mengukur kebijakan kapasitas; Brier dan calibration mengukur kualitas probabilitas."),
        ("Bab 3. Metode", "Dataset UCI bank-additional-full berisi 41.188 observasi, 20 input dan y. Raw dipertahankan, duplikasi dicatat tanpa deduplikasi otomatis. Core_v1 memakai age, job, marital, education, default, housing, loan, pdays, previous, poutcome. Duration dan campaign dilarang. pdays=999 menjadi flag previously_contacted=0 dan pdays_value=0; pdays=0 tetap flag=1. OHE/scaler hanya fit pada train/fold. Train [0,24712), calibration [24712,28831), selection [28831,32950), test [32950,41188). TimeSeriesSplit tiga fold expanding-window memakai gap=0. Urutan adalah proksi waktu, bukan tanggal kalender. Grid 41 kandidat, 123 fold-fits; sigmoid melalui FrozenEstimator hanya fit calibration. Threshold dipilih F1 selection; tie threshold lebih tinggi. Winner AP selection practical tie <=0.005, lalu precision@10%, Brier, latency, model ID. Lock dibuat sebelum test; model tidak di-refit."),
        ("Bab 4. Implementasi dan evaluasi", "Python 3.11, scikit-learn, Streamlit, JSON/CSV/joblib registry lokal. Dashboard tidak training pada rerun. Scoring tervalidasi all-or-nothing; ranking tie p_yes menurut input_row_id; K dan threshold terpisah. Tabel di bawah berasal dari run tercatat. Dummy prior bukan model keenam; random ranking bukan probabilitas. Model test terbaik tidak mengganti winner selection."),
        ("Bab 5. Kesimpulan dan saran", f"Winner deployment baseline tetap {lock['winner_model_id']}, dipilih sebelum hasil test dibuka. Hasil dapat buruk dan tetap sah secara metodologis. Prevalensi train sekitar 4,81% dan test sekitar 30,83% menandakan drift besar; kalibrasi tidak dijamin stabil. Dataset Portugal 2008-2010 tidak otomatis berlaku di Indonesia saat ini. Tanpa ID unik, as-of dates, consent, biaya dan margin, hasil bukan efek kausal atau ROI aktual. Riset berikutnya: holdout baru, snapshot as-of, customer grouping jika ID tersedia, drift monitoring, subgroup audit dan calibration refresh dengan protokol baru."),
        ("Bab 6. Lampiran", f"Sumber dataset: {SOURCE}\nUnduhan dataset publik: {DATA_URL}\nSitasi: Moro, Cortez dan Rita (2014), A Data-Driven Approach to Predict the Success of Bank Telemarketing, Decision Support Systems, DOI 10.1016/j.dss.2014.03.001.\nTautan kode publik: BELUM DIPUBLIKASIKAN.\nTautan slide publik: BELUM DIPUBLIKASIKAN.\nVideo demo: BELUM DIREKAM.\nIdentitas anggota dan konfirmasi rubrik NLP: MENUNGGU TIM/DOSEN.\nReproduksi: README.md, configs/experiment.yaml, requirements.lock dan direktori run. Tidak ada publikasi otomatis.")]
    for title, text in chapters:
        add(title, "Heading1")
        add(text)
        if title.startswith("Bab 3"):
            add(f"Dataset SHA-256: {manifest['dataset_sha256']}\nConfig SHA-256: {manifest['config_sha256']}\nLock SHA-256: {lock['lock_sha256']}")
        if title.startswith("Bab 4"):
            add("Validation-selection (winner dipilih di sini)", "Heading2")
            table(selection)
            if test is not None:
                add("Final test (tidak dipakai memilih winner)", "Heading2")
                table(test)
                winner_report = read_json(run / "evaluations" / f"test_{lock['winner_model_id']}.json")
                add(f"Confusion matrix labels [0,1]: {winner_report['confusion_matrix']}. Threshold terkunci {winner_report['metrics']['threshold']:.2f}.")
                for kind in ("pr", "calibration", "capacity"):
                    image = run / "charts" / f"test_{lock['winner_model_id']}_{kind}.png"
                    story.extend([Image(str(image), width=450, height=277), Spacer(1, 12)])
            else:
                add("Final test belum dievaluasi; selection tidak disebut hasil test.")
            add("Parameter terpilih dan bukti runtime", "Heading2")
            for model_id in lock["candidate_model_hashes"]:
                metadata = read_json(run / "models" / model_id / "model_metadata.json")
                add(f"{model_id}: {metadata['hyperparameters']}; inner-CV AP={metadata['best_inner_cv_ap']:.4f}; fit={metadata['fit_seconds']:.3f}s; threshold={metadata['threshold']:.2f}.")
            if (run / "benchmark.json").exists():
                measured = read_json(run / "benchmark.json")
                add(f"Perangkat aktual: {measured['environment']['cpu_count']} logical CPU, RAM {measured['environment']['ram_bytes'] / 1024**3:.2f} GiB. Grid training {measured['training_grid_seconds']:.3f}s; registry disk hangat {measured['startup_registry_warmed_seconds']:.3f}s; single median 20 repetisi {measured['single_median_seconds_20_repetitions']:.3f}s; parse/validate/predict/rank 10.000 baris {measured['batch_parse_validate_predict_rank_seconds']:.3f}s; peak RSS tersampling {measured['peak_process_rss_bytes_sampled'] / 1024**2:.1f} MiB. Tidak mengklaim pengukuran pada acuan 4-core/8GB atau cold startup Streamlit.")
            add("Bukti software tests tersedia di artifacts/software_test_results.xml; screenshots/rehearsal video masih perlu review tim. File audit/split dan metadata menghubungkan setiap angka ke dataset/config/environment.")
        story.append(PageBreak())
    SimpleDocTemplate(str(reports / "final_report.pdf"), pagesize=A4, leftMargin=35, rightMargin=35, title="DepoRank - Draft laporan enam bab").build(story)
    slide_story = []
    slides = [("DepoRank / keputusan kampanye", "Prioritaskan observasi kontak pada kapasitas terbatas. Prototipe akademik; bukan identitas nasabah unik."),
              ("Data / provenance", f"41.188 observasi UCI additional-full. Raw immutable; hash {manifest['dataset_sha256']}."),
              ("KDD / tanpa leakage", "10 fitur core; buang duration/campaign. pdays sentinel, OHE/scaler fold train, lima model nyata."),
              ("Validasi / keputusan dikunci", "Expanding-window train; sigmoid calibration terpisah; selection menentukan winner/threshold; test sekali sesudah lock."),
              ("Hasil / winner pra-test", f"{lock['winner_model_id']}\n" + selection[["model_id", "average_precision", "precision_at_k"]].to_string(index=False)),
              ("Demo / input sampai ekspor", "Data dan KDD; comparison; prediksi tunggal; batch valid q=10%; batch invalid; ekspor CSV; manifest. Jangan training saat demo."),
              ("Batas / keputusan bertanggung jawab", "Drift besar; Portugal2008-2010; tanpa ID/as-of/consent/biaya. Probabilitas bukan kepastian, ranking bukan uplift kausal. Nama/NIM, persetujuan, durasi, tautan dan video masih pending.")]
    if test is not None:
        slides.insert(5, ("Final test / bukan pemilihan ulang", test[["model_id", "average_precision", "precision_at_k"]].to_string(index=False)))
    for title, text in slides:
        slide_story.extend([paragraph(title, "Title"), Spacer(1, 24), paragraph(text), PageBreak()])
    SimpleDocTemplate(str(presentation / "slides.pdf"), pagesize=landscape(A4), title="DepoRank - Draft stakeholder slides").build(slide_story)
    proposal = [paragraph("DepoRank - Draft Proposal", "Title")]
    for title, text in chapters[:3]:
        proposal.extend([paragraph(title, "Heading1"), paragraph(text), Spacer(1, 12)])
    proposal.extend([paragraph("Output dan rencana", "Heading1"), paragraph("Audit/provenance, lima model+baseline, lock/test, dashboard, ranking dan ekspor. Tahap data, ML, freeze/test, UI, QA dan paket akademik mengikuti SRD. Pendekatan tabular; rubrik proposed NLP approach wajib dikonfirmasi dosen, tanpa menambahkan NLP tidak relevan. Identitas, jadwal dan persetujuan belum tersedia.")])
    SimpleDocTemplate(str(reports / "proposal.pdf"), pagesize=A4, title="DepoRank - Draft Proposal").build(proposal)
    summary = {"run_id": run_id, "winner_model_id": lock["winner_model_id"], "manifest": str(run / "run_manifest.json"),
               "selection_lock_sha256": lock["lock_sha256"], "feature_set": lock["feature_set"],
               "status": manifest["status"], "charts": len(chart_paths), "academic_status": "draft_pending_identity_approval_public_links_video"}
    write_json(reports / "experiment_summary.json", summary)
    return {**summary, "report": "reports/final_report.pdf", "slides": "presentation/slides.pdf", "proposal": "reports/proposal.pdf"}

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bank_marketing.storage import read_json


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
                ax.plot([value["mean_predicted"] for value in values], [value["observed"] for value in values], "o-", color=color)
                ax.plot([0, 1], [0, 1], "--", color="gray")
                ax.set(xlabel="Mean predicted probability", ylabel="Observed yes proportion")
            elif kind == "capacity":
                values = [value for value in curves["capacity"] if "policy" not in value]
                for metric in ("precision_at_k", "recall_at_k", "lift_at_k"):
                    ax.plot([value["fraction"] for value in values], [value[metric] for value in values], "o-", label=metric)
                ax.set(xlabel="Contact capacity fraction", ylabel="Metric value")
                ax.legend(fontsize=8)
            else:
                cm = report["confusion_matrix"]
                ax.imshow(cm, cmap="Blues")
                for row_index in range(2):
                    for column_index in range(2):
                        ax.text(column_index, row_index, str(cm[row_index][column_index]), ha="center", va="center")
                ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no", "yes"], yticklabels=["no", "yes"], xlabel="Predicted", ylabel="Actual")
            ax.set_title(f"{report['split']} / {report['model_id']} / {kind}", fontsize=11)
            fig.tight_layout()
            path = chart_dir / f"{report['split']}_{report['model_id']}_{kind}.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            output.append(str(path))
    return output


def generate_reports(run_id, root="artifacts"):
    from bank_marketing.academic import build_report
    from bank_marketing.ml.registry import Registry

    registry = Registry(root)
    registry.verify_run(run_id)
    render_charts(registry.run_path(run_id))
    return build_report(run_id, root)

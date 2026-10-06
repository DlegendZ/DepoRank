from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from bank_marketing.ml.evaluate import evaluate_scores, positive_probabilities
from bank_marketing.ml.registry import Registry
from bank_marketing.ml.train import load_run_data
from bank_marketing.storage import read_json, utc_now, write_json


def interpret_selection(run_id, root="artifacts"):
    registry = Registry(root)
    ref = registry.load(run_id)
    path = registry.run_path(run_id)
    data, _ = load_run_data(path)
    ranges = read_json(path / "split_manifest.json")["ranges"]
    selection = data.iloc[slice(*ranges["selection"])].reset_index(drop=True)
    X = selection[ref.metadata["raw_feature_names"]]
    y = selection.y.map({"no": 0, "yes": 1}).to_numpy()
    p = positive_probabilities(ref.model, X)
    importance = permutation_importance(ref.model, X, y, scoring="average_precision", n_repeats=5, random_state=42, n_jobs=1)
    features = [{"feature": name, "mean_ap_decrease": float(mean), "std_ap_decrease": float(std),
                 "repeats": [float(value) for value in values]} for name, mean, std, values in
                zip(X.columns, importance.importances_mean, importance.importances_std, importance.importances)]
    order = np.lexsort((np.arange(len(p)), -p))
    chosen = np.zeros(len(p), dtype=bool)
    chosen[order[:math.ceil(0.1 * len(p))]] = True
    groups = []
    subgroup_data = X.copy()
    subgroup_data["age_group"] = pd.cut(subgroup_data.age, [0, 25, 40, 60, 120], labels=["1-25", "26-40", "41-60", "61-120"])
    for column in ("age_group", "job", "marital", "education"):
        for value in sorted(subgroup_data[column].dropna().unique(), key=str):
            mask = (subgroup_data[column] == value).to_numpy()
            evaluation = evaluate_scores(y[mask], p[mask], ref.threshold)
            contacted = int(chosen[mask].sum())
            hits = int(y[mask][chosen[mask]].sum())
            groups.append({"column": column, "group": str(value), "n": int(mask.sum()),
                           "insufficient_evidence": int(mask.sum()) < 100, "selection_rate": float(chosen[mask].mean()),
                           "precision_at_global_capacity": hits / contacted if contacted else None,
                           "recall_at_global_capacity": hits / int(y[mask].sum()) if y[mask].sum() else None,
                           "metrics": evaluation["metrics"], "calibration": evaluation["curves"]["calibration"]})
    report = {"run_id": run_id, "model_id": ref.model_id, "split": "selection", "range": ranges["selection"],
              "created_at_utc": utc_now(), "permutation_importance": features, "subgroups": groups,
              "notes": ["Importance includes negative values; correlated features limit interpretation. These are not causal effects.",
                        "Subgroups with n<100 have insufficient evidence. Limited metrics do not establish fairness.",
                        "Only selection data is used; features, winner, thresholds, and test baseline remain unchanged."]}
    write_json(path / "interpretation.json", report)
    return report

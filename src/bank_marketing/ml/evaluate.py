from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from bank_marketing.schemas import DomainError


def positive_probabilities(model, data) -> np.ndarray:
    """Extract the probability column whose fitted class is numeric 1."""
    if model is None or not callable(getattr(model, "predict_proba", None)):
        raise DomainError("MODEL_NOT_READY", "A fitted probabilistic model is required.")
    classes = np.asarray(getattr(model, "classes_", []))
    positive = np.flatnonzero(classes == 1)
    if classes.ndim != 1 or len(positive) != 1:
        raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Model must expose exactly one positive class 1.")
    try:
        probabilities = np.asarray(model.predict_proba(data), dtype=float)
    except Exception as exc:
        raise DomainError("INTERNAL_ERROR", "Model inference failed; no rows were scored.") from exc
    if probabilities.ndim != 2 or probabilities.shape != (len(data), len(classes)):
        raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Probability output shape does not match model classes and input rows.")
    p_yes = probabilities[:, positive[0]]
    if not np.isfinite(p_yes).all() or ((p_yes < 0) | (p_yes > 1)).any():
        raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Positive probabilities must be finite and between 0 and 1.")
    return p_yes


def _validated_scores(y, p, row_ids=None):
    try:
        y, p = np.asarray(y), np.asarray(p)
        valid = (y.ndim == p.ndim == 1 and len(y) > 0 and len(y) == len(p)
                 and y.dtype.kind in "iuf" and p.dtype.kind in "iuf"
                 and np.isin(y, [0, 1]).all() and np.isfinite(p).all()
                 and ((p >= 0) & (p <= 1)).all())
        ids = np.arange(len(y)) if row_ids is None else np.asarray(row_ids)
        valid = valid and (ids.ndim == 1 and len(ids) == len(y) and ids.dtype.kind in "iu"
                           and (ids >= 0).all() and len(np.unique(ids)) == len(ids))
    except (TypeError, ValueError, OverflowError) as error:
        raise DomainError("ROW_INVALID", "Data evaluasi/probabilitas/row ID tidak valid.") from error
    if not valid:
        raise DomainError("ROW_INVALID", "Data evaluasi/probabilitas/row ID tidak valid.")
    return y.astype(int), p.astype(float), ids


def _unit_interval(value, code):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or not 0 <= value <= 1:
        raise DomainError(code, "Nilai harus numeric finite antara 0 dan 1.")


def threshold_for_f1(y, p) -> float:
    y, p, _ = _validated_scores(y, p)
    values = [(f1_score(y, p >= t, zero_division=0), float(t))
              for t in np.linspace(0, 1, 101)]
    return max(values)[1]


def capacity_metrics(y, p, k, row_ids=None) -> dict:
    y, p, ids = _validated_scores(y, p, row_ids)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 0 <= k <= len(y):
        raise DomainError("CAPACITY_INVALID", "K harus integer antara 0 dan jumlah baris.")
    order = np.lexsort((ids, -p))
    total, prevalence = int(y.sum()), float(y.mean()) if len(y) else 0.0
    hits = int(y[order[:k]].sum())
    precision = hits / k if k else None
    return {"capacity_k": int(k), "precision_at_k": precision,
            "recall_at_k": hits / total if total and k else None,
            "lift_at_k": precision / prevalence if k and prevalence else None,
            "reason_codes": {**({"precision_at_k": "ZERO_CAPACITY", "recall_at_k": "ZERO_CAPACITY", "lift_at_k": "ZERO_CAPACITY"} if not k else {}),
                             **({"recall_at_k": "NO_POSITIVES", "lift_at_k": "NO_POSITIVES"} if not total else {})}}


def evaluate_scores(y, p, threshold=0.5, row_ids=None, q=None, *, k=None) -> dict:
    y, p, row_ids = _validated_scores(y, p, row_ids)
    _unit_interval(threshold, "ROW_INVALID")
    if q is not None and k is not None:
        raise DomainError("CAPACITY_INVALID", "Berikan hanya satu mode kapasitas: K atau q.")
    if q is not None:
        _unit_interval(q, "CAPACITY_INVALID")
    capacity_k = k if k is not None else math.ceil((0.1 if q is None else q) * len(y))
    cap = capacity_metrics(y, p, capacity_k, row_ids)
    pred = (p >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    reasons = {} if both else {"roc_auc": "SINGLE_CLASS", "average_precision": "SINGLE_CLASS"}
    if pred.sum() == 0:
        reasons.update({"precision": "NO_PREDICTED_POSITIVES", "f1_warning": "NO_PREDICTED_POSITIVES"})
    metrics = {"n": len(y), "prevalence": float(y.mean()), "threshold": float(threshold),
               "average_precision": float(average_precision_score(y, p)) if both else None,
               "roc_auc": float(roc_auc_score(y, p)) if both else None,
               "precision": float(precision_score(y, pred, zero_division=0)),
               "recall": float(recall_score(y, pred, zero_division=0)) if y.sum() else None,
               "f1": float(f1_score(y, pred, zero_division=0)),
               "accuracy": float(accuracy_score(y, pred)), "brier_score": float(brier_score_loss(y, p)),
               **{key: value for key, value in cap.items() if key != "reason_codes"},
               "reason_codes": {**reasons, **cap["reason_codes"]}}
    if not y.sum():
        metrics["reason_codes"]["recall"] = "NO_POSITIVES"
    pr, rc, pt = precision_recall_curve(y, p) if both else ([], [], [])
    fpr, tpr, rt = roc_curve(y, p) if both else ([], [], [])
    calibration = []
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    bins = np.searchsorted(edges[1:-1], p, side="right")
    for b in np.unique(bins):
        mask = bins == b
        calibration.append({"mean_predicted": float(p[mask].mean()), "observed": float(y[mask].mean()), "count": int(mask.sum())})
    capacity = []
    for fraction in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0):
        capacity.append({"fraction": fraction, **capacity_metrics(y, p, math.ceil(fraction * len(y)), row_ids)})
    if len(y) >= 1000:
        capacity.append({"fraction": 1000 / len(y), **capacity_metrics(y, p, 1000, row_ids), "policy": "explicit_k_1000"})
    clean = lambda xs: [float(x) if np.isfinite(x) else None for x in xs]
    return {"metrics": metrics, "confusion_matrix": cm, "curves": {
        "precision_recall": {"precision": clean(pr), "recall": clean(rc), "thresholds": clean(pt)},
        "roc": {"fpr": clean(fpr), "tpr": clean(tpr), "thresholds": clean(rt)},
        "calibration": calibration, "capacity": capacity}}

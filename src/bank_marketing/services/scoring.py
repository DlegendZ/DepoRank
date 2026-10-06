"""Validated all-or-nothing scoring using registry-supplied fitted models."""

from datetime import datetime, timezone
from math import isfinite
from numbers import Real
from uuid import uuid4

import numpy as np
import pandas as pd

from bank_marketing.data.audit import audit_dataset, normalize_records
from bank_marketing.ml.evaluate import positive_probabilities
from bank_marketing.schemas import CORE_FEATURES, DomainError


def score_records(records: pd.DataFrame, model_ref, mode: str = "scoring") -> dict:
    """Validate the whole batch before inference; demo explicitly discards non-core inputs."""
    if model_ref is None or getattr(model_ref, "model", None) is None:
        raise DomainError("MODEL_NOT_READY", "Prepare and activate a locked model before scoring.")
    if mode not in {"scoring", "demo"}:
        raise DomainError("SCHEMA_INVALID", "Scoring mode must be scoring or demo.")
    if not isinstance(records, pd.DataFrame) or records.empty:
        raise DomainError("SCHEMA_INVALID", "Scoring requires a nonempty DataFrame.")
    if len(records) > 50000:
        raise DomainError("SCHEMA_INVALID", "Scoring accepts at most 50,000 rows.")
    metadata = getattr(model_ref, "metadata", None)
    feature_set = getattr(model_ref, "feature_set", None)
    if not isinstance(metadata, dict):
        raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Model metadata is required.")
    raw_names = metadata.get("raw_feature_names")
    if feature_set != "core_v1" or not isinstance(raw_names, (list, tuple)) or len(raw_names) != len(CORE_FEATURES) or set(raw_names) != set(CORE_FEATURES):
        raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Scoring requires core_v1 metadata with the ten raw core features.")
    threshold = getattr(model_ref, "threshold", None)
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not isfinite(threshold) or not 0 <= threshold <= 1:
        raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Model threshold must be finite and between 0 and 1.")
    if not getattr(model_ref, "run_id", None) or not getattr(model_ref, "model_id", None):
        raise DomainError("MODEL_NOT_READY", "Model must identify its run and model version.")
    report = audit_dataset(records, mode=mode)
    if not report["valid"]:
        errors = report.get("errors", [])
        schema_invalid = any(error.get("code") == "SCHEMA_INVALID" for error in errors)
        raise DomainError("SCHEMA_INVALID" if schema_invalid else "ROW_INVALID", "Batch validation failed; no rows were scored.", details={"errors": errors, "warnings": report.get("warnings", [])})
    normalized = normalize_records(records).reset_index(drop=True)
    warnings = list(report.get("warnings", []))
    if mode == "demo":
        discarded = [column for column in records.columns if column not in {*CORE_FEATURES, "record_id"}]
        for column in discarded:
            if not any(isinstance(warning, dict) and warning.get("column") == column and warning.get("code") == "DEMO_COLUMN_DISCARDED" for warning in warnings):
                warnings.append({"code": "DEMO_COLUMN_DISCARDED", "column": column, "input_row_id": None, "message": "Demo input column explicitly discarded before inference."})
    features = normalized.loc[:, list(raw_names)]
    categories = metadata.get("training_categories", {})
    if not isinstance(categories, dict):
        raise DomainError("ARTIFACT_INTEGRITY_ERROR", "training_categories must map columns to fitted categories.")
    for column, seen in categories.items():
        if column in features:
            for row_id, value in enumerate(features[column]):
                if value not in seen:
                    warnings.append({"code": "UNSEEN_CATEGORY", "input_row_id": row_id, "column": column, "message": "Valid category was not present in fitted training data."})
    p_yes = positive_probabilities(model_ref.model, features)
    batch_id = uuid4().hex
    row_warnings = [[] for _ in range(len(records))]
    for warning in warnings:
        row_id = warning.get("input_row_id") if isinstance(warning, dict) else None
        if isinstance(row_id, (int, np.integer)) and 0 <= row_id < len(records):
            row_warnings[row_id].append(warning)
    record_ids = normalized["record_id"].tolist() if "record_id" in normalized else [f"{batch_id}:{row_id}" for row_id in range(len(records))]
    return {
        "batch_id": batch_id,
        "run_id": model_ref.run_id,
        "model_id": model_ref.model_id,
        "feature_set": feature_set,
        "threshold": float(threshold),
        "model_hash": getattr(model_ref, "model_hash", None),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": [
            {"input_row_id": row_id, "record_id": record_ids[row_id], "p_yes": float(p), "predicted_label": "yes" if p >= threshold else "no", "warnings": row_warnings[row_id]}
            for row_id, p in enumerate(p_yes)
        ],
        "warnings": warnings,
    }

"""In-memory exports with strict JSON and spreadsheet-safe CSV strings."""

import csv
import io
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from numbers import Integral, Real

import numpy as np
import pandas as pd

from bank_marketing.schemas import DomainError


RESULT_COLUMNS = (
    "batch_id", "input_row_id", "record_id", "rank", "p_yes", "predicted_label",
    "selected_for_contact", "model_id", "run_id", "feature_set", "threshold", "capacity_k",
)
ERROR_COLUMNS = ("input_row_id", "column", "code", "message")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def _json_bytes(value) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, default=_json_default).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise DomainError("SCHEMA_INVALID", "Export requires finite JSON-compatible values.") from exc


def _safe_cell(value):
    if value is None:
        return ""
    if isinstance(value, (Mapping, list, tuple)):
        value = _json_bytes(value).decode("utf-8")
    if isinstance(value, str):
        prefix = 0
        while prefix < len(value) and (value[prefix].isspace() or unicodedata.category(value[prefix]) in {"Cc", "Cf"}):
            prefix += 1
        if prefix < len(value) and value[prefix] in "=+-@":
            return "'" + value
        return value
    if isinstance(value, Real) and not isinstance(value, Integral) and not math.isfinite(value):
        raise DomainError("SCHEMA_INVALID", "CSV export cannot contain NaN or Infinity.")
    return value


def _csv_bytes(columns, rows) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerow([_safe_cell(column) for column in columns])
    for row in rows:
        writer.writerow([_safe_cell(row.get(column)) for column in columns])
    return output.getvalue().encode("utf-8-sig")


def _feature_rows(include_features, rows):
    if include_features is None:
        return [], {}
    if isinstance(include_features, pd.DataFrame):
        frame = include_features.copy()
        if frame.columns.duplicated().any():
            raise DomainError("SCHEMA_INVALID", "Feature export columns must be unique.")
        if "input_row_id" in frame:
            ids = frame.pop("input_row_id").tolist()
        else:
            ids = list(range(len(frame)))
        if any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise DomainError("SCHEMA_INVALID", "Feature row IDs must be unique nonnegative integers.")
        features = dict(zip(ids, frame.to_dict("records")))
        columns = frame.columns.tolist()
        if any(row["input_row_id"] not in features for row in rows):
            raise DomainError("SCHEMA_INVALID", "Feature export must cover every exported input_row_id.")
    elif isinstance(include_features, Sequence) and not isinstance(include_features, (str, bytes)):
        columns = list(include_features)
        features = {row["input_row_id"]: {name: row.get(name) for name in columns} for row in rows}
    else:
        raise DomainError("SCHEMA_INVALID", "include_features must be a DataFrame or list of row field names.")
    if any(not isinstance(name, str) or name in RESULT_COLUMNS for name in columns) or len(set(columns)) != len(columns):
        raise DomainError("SCHEMA_INVALID", "Feature columns cannot overwrite generated result columns.")
    return columns, features


def export_results(result: dict, format: str = "csv", top_k_only: bool = False, include_features=None) -> bytes:
    """Export all rows by default; optional features map through input_row_id, not rank."""
    if format not in {"csv", "json"}:
        raise DomainError("SCHEMA_INVALID", "Export format must be csv or json.")
    if not isinstance(top_k_only, bool):
        raise DomainError("SCHEMA_INVALID", "top_k_only must be boolean.")
    if not isinstance(result, dict) or not isinstance(result.get("rows"), list):
        raise DomainError("SCHEMA_INVALID", "Export requires a result with rows.")
    rows = result["rows"]
    if top_k_only:
        if any("selected_for_contact" not in row for row in rows):
            raise DomainError("SCHEMA_INVALID", "Top-K export requires ranked results.")
        rows = [row for row in rows if row["selected_for_contact"]]
    columns, features = _feature_rows(include_features, rows)
    exported = []
    for row in rows:
        values = {name: result.get(name) for name in RESULT_COLUMNS}
        values.update({name: row.get(name, values[name]) for name in RESULT_COLUMNS})
        values.update(features.get(row["input_row_id"], {}))
        exported.append(values)
    if format == "json":
        payload = dict(result)
        payload["rows"] = [
            {**{name: row[name] for name in (*RESULT_COLUMNS, "warnings") if name in row},
             **features.get(row["input_row_id"], {})}
            for row in rows
        ]
        return _json_bytes(payload)
    return _csv_bytes([*RESULT_COLUMNS, *columns], exported)


def export_errors(errors) -> bytes:
    """Export audit errors without raw data or tracebacks."""
    if isinstance(errors, dict):
        errors = errors.get("errors", [])
    if not isinstance(errors, list) or any(not isinstance(error, dict) for error in errors):
        raise DomainError("SCHEMA_INVALID", "Error export requires a list of error mappings.")
    return _csv_bytes(ERROR_COLUMNS, errors)


def export_metadata(result: dict) -> bytes:
    """Export batch provenance without suggesting new performance measurements."""
    rows = result.get("rows", [])
    metadata = {key: result.get(key) for key in (
        "batch_id", "run_id", "model_id", "model_hash", "feature_set", "threshold",
        "created_at_utc", "capacity_k", "call_fraction",
    )}
    metadata["created_at_utc"] = metadata["created_at_utc"] or datetime.now(timezone.utc).isoformat()
    metadata["batch_size"] = len(rows)
    metadata["warning_count"] = len(result.get("warnings", []))
    metadata["policy"] = {"invalid_policy": "reject_batch", "ranking": "p_yes_desc_input_row_id_asc",
                          "selection": "exact_top_capacity_k", "label": "p_yes_greater_equal_locked_threshold"}
    return _json_bytes(metadata)

"""Strict row-level audit; normalization never changes source records."""

import numpy as np
import pandas as pd

from bank_marketing.schemas import (
    CATEGORIES, CORE_FEATURES, INTEGER_COLUMNS, NUMERIC_COLUMNS,
    TRAINING_COLUMNS, DomainError,
)


def normalize_records(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise DomainError("SCHEMA_INVALID", "Records must be a DataFrame.")
    result = data.copy(deep=True)
    for column in result.columns:
        if result[column].dtype == object or isinstance(result[column].dtype, pd.StringDtype):
            result[column] = result[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    return result


def _issue(row, column, code, message):
    return {"input_row_id": row, "column": column, "code": code, "message": message}


def audit_dataset(data: pd.DataFrame, mode="training") -> dict:
    if mode not in {"training", "scoring", "demo"}:
        raise DomainError("SCHEMA_INVALID", "Audit mode must be training, scoring or demo.")
    if not isinstance(data, pd.DataFrame):
        raise DomainError("SCHEMA_INVALID", "Records must be a DataFrame.")
    errors, warnings = [], []
    columns = list(data.columns)
    report = {
        "valid": False, "errors": errors, "warnings": warnings,
        "counts": {"rows": len(data), "columns": len(columns)},
        "schema": {"mode": mode, "columns": columns, "dtypes": {}},
        "descriptive": {}, "summary": {},
    }
    if data.columns.has_duplicates or not all(isinstance(name, str) for name in columns):
        errors.append(_issue(None, None, "SCHEMA_INVALID", "Column names must be unique strings."))
        return report
    expected = set(TRAINING_COLUMNS if mode == "training" else CORE_FEATURES)
    allowed = expected | ({"record_id"} if mode != "training" else set())
    if mode == "demo":
        expected = set(CORE_FEATURES) | {"y"}
        allowed = set(TRAINING_COLUMNS) | {"record_id"}
        actual = set(columns) - {"record_id"}
        if actual not in (set(TRAINING_COLUMNS), expected):
            errors.append(_issue(None, None, "SCHEMA_INVALID", "Demo requires all 21 training columns or ten core columns plus y."))
    for column in sorted(expected - set(columns)):
        errors.append(_issue(None, column, "SCHEMA_INVALID", "Required column is missing."))
    for column in sorted(set(columns) - allowed):
        errors.append(_issue(None, column, "SCHEMA_INVALID", "Unexpected column is forbidden in this mode."))
    if data.empty:
        errors.append(_issue(None, None, "SCHEMA_INVALID", "Records must contain at least one row."))
    report["schema"]["dtypes"] = {column: str(dtype) for column, dtype in data.dtypes.items()}
    if errors:
        return report
    normalized = normalize_records(data)
    for column in columns:
        nonscalar = normalized[column].map(lambda value: not pd.api.types.is_scalar(value))
        errors.extend(_issue(int(row), column, "ROW_INVALID", "Record values must be scalar, not nested objects.")
                      for row in np.flatnonzero(nonscalar))
    if errors:
        report["counts"].update({"errors": len(errors), "warnings": len(warnings)})
        return report
    if mode == "demo":
        warnings.extend(_issue(None, column, "DEMO_COLUMN_DISCARDED", "Demo adapter explicitly excludes this column from model inputs.")
                        for column in columns if column not in {*CORE_FEATURES, "record_id"})
    numeric = {}
    for column in columns:
        values = normalized[column].reset_index(drop=True)
        original = data[column].reset_index(drop=True)
        changed = original.map(lambda value: isinstance(value, str) and value != value.strip())
        warnings.extend(_issue(int(row), column, "WHITESPACE_NORMALIZED", "Outer whitespace normalized on a copy.") for row in np.flatnonzero(changed))
        missing = values.isna() | values.map(lambda value: isinstance(value, str) and value == "")
        errors.extend(_issue(int(row), column, "ROW_INVALID", "Blank or missing value is not allowed.") for row in np.flatnonzero(missing))
        if column in CATEGORIES:
            invalid = ~missing & ~values.isin(CATEGORIES[column])
            code = "SCHEMA_INVALID" if column == "y" else "ROW_INVALID"
            errors.extend(_issue(int(row), column, code, "Value is not an allowed category.") for row in np.flatnonzero(invalid))
        elif column in NUMERIC_COLUMNS:
            invalid_type = values.map(lambda value: isinstance(value, (bool, np.bool_, complex)) or not isinstance(value, (str, int, float, np.integer, np.floating)))
            converted = pd.to_numeric(values.mask(invalid_type), errors="coerce")
            array = converted.to_numpy(dtype=float, na_value=np.nan)
            invalid = ~missing.to_numpy() & (invalid_type.to_numpy() | ~np.isfinite(array))
            errors.extend(_issue(int(row), column, "ROW_INVALID", "Numeric value must be finite and valid, not boolean.") for row in np.flatnonzero(invalid))
            good = ~missing.to_numpy() & ~invalid
            if column in INTEGER_COLUMNS:
                fractional = good & (array != np.floor(array))
                errors.extend(_issue(int(row), column, "ROW_INVALID", "Value must be integral.") for row in np.flatnonzero(fractional))
                good &= ~fractional
            limits = {"age": (1, 120), "pdays": (0, 999), "previous": (0, np.inf), "duration": (0, np.inf), "campaign": (1, np.inf)}
            if column in limits:
                low, high = limits[column]
                outside = good & ((array < low) | (array > high))
                errors.extend(_issue(int(row), column, "ROW_INVALID", "Value is outside the allowed range.") for row in np.flatnonzero(outside))
                good &= ~outside
            historical = {"age": (17, 98), "pdays": (0, 27), "previous": (0, 7)}
            if column in historical:
                low, high = historical[column]
                outside = good & ((array < low) | (array > high))
                if column == "pdays":
                    outside &= array != 999
                warnings.extend(_issue(int(row), column, "OUTSIDE_HISTORICAL_RANGE", "Valid value lies outside the official historical snapshot range.") for row in np.flatnonzero(outside))
            numeric[column] = converted
            finite = array[np.isfinite(array)]
            magnitude = float(np.max(np.abs(finite))) if len(finite) else 0.0
            # Scaling keeps finite extrema from overflowing intermediate sums.
            scaled = finite / magnitude if magnitude else finite
            report["descriptive"][column] = {
                "count": int(len(finite)), "min": float(finite.min()) if len(finite) else None,
                "max": float(finite.max()) if len(finite) else None,
                "mean": float(scaled.mean() * magnitude) if len(finite) else None,
                "median": float(np.median(scaled) * magnitude) if len(finite) else None,
            }
        elif column == "record_id":
            invalid = ~missing & ~values.map(lambda value: isinstance(value, str))
            errors.extend(_issue(int(row), column, "ROW_INVALID", "record_id must be a string.") for row in np.flatnonzero(invalid))
            duplicate = values.duplicated(keep=False) & ~missing
            errors.extend(_issue(int(row), column, "ROW_INVALID", "record_id must be unique after whitespace normalization.") for row in np.flatnonzero(duplicate))
    if "pdays" in numeric and "previous" in numeric:
        inconsistent = (numeric["pdays"] == 999) & (numeric["previous"] > 0)
        warnings.extend(_issue(int(row), "pdays", "SEMANTIC_INCONSISTENCY", "pdays=999 with previous>0; values retained unchanged.") for row in np.flatnonzero(inconsistent))
        inconsistent = (numeric["previous"] == 0) & (normalized["poutcome"].reset_index(drop=True) == "success")
        warnings.extend(_issue(int(row), "poutcome", "SEMANTIC_INCONSISTENCY", "previous=0 with poutcome=success; values retained unchanged.") for row in np.flatnonzero(inconsistent))
    report["counts"].update({
        "duplicate_rows": int(normalized.duplicated().sum()),
        "unknown": {column: int((normalized[column] == "unknown").sum()) for column in columns if column in CATEGORIES},
        "target": {label: int((normalized["y"] == label).sum()) for label in ("no", "yes")} if "y" in normalized else None,
        "errors": len(errors), "warnings": len(warnings),
    })
    report["summary"] = {"numeric": report["descriptive"], "categories": {
        column: {label: int((normalized[column] == label).sum()) for label in CATEGORIES[column]}
        for column in columns if column in CATEGORIES
    }}
    report["valid"] = not errors
    return report


def adapt_demo_records(data: pd.DataFrame) -> dict:
    report = audit_dataset(data, mode="demo")
    if not report["valid"]:
        raise DomainError("SCHEMA_INVALID", "Labeled demo validation failed.", report["errors"])
    normalized = normalize_records(data)
    columns = (["record_id"] if "record_id" in normalized else []) + list(CORE_FEATURES)
    return {"records": normalized.loc[:, columns].copy(), "labels": normalized["y"].map({"no": 0, "yes": 1}),
            "dropped_columns": [column for column in data.columns if column not in columns], "warnings": report["warnings"]}

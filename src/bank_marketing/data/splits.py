"""Original-position temporal-proxy splits and expanding-window folds."""

from copy import deepcopy
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from bank_marketing.schemas import SPLIT_RANGES, DomainError


def _targets(data):
    if not isinstance(data, pd.DataFrame) or "y" not in data or data.columns.has_duplicates:
        raise DomainError("SCHEMA_INVALID", "Split data requires unique columns and y.")
    if not data["y"].isin(["no", "yes"]).all():
        raise DomainError("SCHEMA_INVALID", "Split target must contain only no/yes.")
    return data["y"].map({"no": 0, "yes": 1}).to_numpy()


def create_splits(data, config) -> dict:
    y = _targets(data)
    if not isinstance(config, dict) or not isinstance(config.get("split"), dict):
        raise DomainError("SCHEMA_INVALID", "Split configuration is required.")
    split = config["split"]
    if split.get("bounds") != "half_open" or len(data) != 41188:
        raise DomainError("DATASET_VARIANT_MISMATCH", "Fixed splits require 41,188 original ordered rows and half-open bounds.")
    ranges = {}
    for name, expected in SPLIT_RANGES.items():
        bounds = split.get(name)
        if not isinstance(bounds, list) or len(bounds) != 2 or any(type(value) is not int for value in bounds) or bounds != expected:
            raise DomainError("SCHEMA_INVALID", "Split boundaries must match the fixed SRD protocol.", {"split": name})
        ranges[name] = list(bounds)
    if "row_id" in data:
        ids = data["row_id"].to_numpy()
        if not np.array_equal(ids, np.arange(len(data))) or data["row_id"].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise DomainError("SCHEMA_INVALID", "row_id must preserve original zero-based row order.")
    counts, row_sets = {}, {}
    feature_columns = [name for name in data.columns if name != "row_id"]
    hashes = pd.util.hash_pandas_object(data.loc[:, feature_columns], index=False).to_numpy()
    for name, (start, end) in ranges.items():
        labels = y[start:end]
        if not np.array_equal(np.unique(labels), [0, 1]):
            raise DomainError("SCHEMA_INVALID", "Each split requires both target classes.", {"split": name})
        counts[name] = {"rows": end - start, "no": int((labels == 0).sum()), "yes": int((labels == 1).sum()),
                        "prevalence": float(labels.mean()), "duplicate_rows": int(data.iloc[start:end][feature_columns].duplicated().sum())}
        row_sets[name] = set(hashes[start:end].tolist())
    overlap = {f"{left}:{right}": len(row_sets[left] & row_sets[right]) for left, right in combinations(ranges, 2)}
    return {"schema_version": "1.0", "ranges": deepcopy(ranges), "counts": counts,
            "bounds": "half_open", "rows": len(data), "coverage": {"start": 0, "end_exclusive": len(data), "disjoint": True},
            "duplicate_overlap": overlap, "order": "original_csv_position", "shuffle": False,
            "warning": "Original row order is a temporal proxy, not unique customer identity or exact calendar time."}


def inner_folds(data, n_splits=3, gap=0) -> list:
    y = _targets(data)
    if type(n_splits) is not int or n_splits < 2 or type(gap) is not int or gap < 0:
        raise DomainError("SCHEMA_INVALID", "TimeSeriesSplit requires integer n_splits>=2 and gap>=0.")
    if len(data) > 24712:
        raise DomainError("SCHEMA_INVALID", "Inner folds must receive train only, at most 24,712 rows.")
    if "row_id" in data and not np.array_equal(data["row_id"].to_numpy(), np.arange(len(data))):
        raise DomainError("SCHEMA_INVALID", "Inner folds require original train row order.")
    try:
        folds = list(TimeSeriesSplit(n_splits=n_splits, gap=gap).split(data))
    except ValueError as exc:
        raise DomainError("SCHEMA_INVALID", "Train is too small for requested temporal folds.") from exc
    for fold, (fit, validation) in enumerate(folds):
        if not np.array_equal(np.unique(y[fit]), [0, 1]) or not np.array_equal(np.unique(y[validation]), [0, 1]):
            raise DomainError("TRAINING_FAILED", "Every fit and validation fold must contain both classes.", {"fold": fold})
    return folds

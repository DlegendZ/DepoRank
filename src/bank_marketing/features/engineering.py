"""Deterministic, stateless feature selection and pdays engineering."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from bank_marketing.data.audit import audit_dataset, normalize_records
from bank_marketing.schemas import CATEGORIES, CORE_FEATURES, CONTEXT_FEATURES, DomainError


def raw_features(feature_set):
    if feature_set == "core_v1":
        return CORE_FEATURES
    if feature_set == "context_v1":
        return CONTEXT_FEATURES
    raise DomainError("SCHEMA_INVALID", "Unknown feature set.")


def engineered_features(feature_set):
    return tuple(name for column in raw_features(feature_set)
                 for name in (("previously_contacted", "pdays_value") if column == "pdays" else (column,)))


class FeatureEngineer(TransformerMixin, BaseEstimator):
    def __init__(self, feature_set="core_v1"):
        self.feature_set = feature_set

    def _select(self, X):
        if not isinstance(X, pd.DataFrame) or X.columns.has_duplicates:
            raise DomainError("SCHEMA_INVALID", "Features require a DataFrame with unique columns.")
        names = raw_features(self.feature_set)
        missing = set(names) - set(X.columns)
        if missing:
            raise DomainError("SCHEMA_INVALID", "Required raw features are missing.", sorted(missing))
        selected = X.loc[:, list(names)].copy()
        report = audit_dataset(selected.loc[:, list(CORE_FEATURES)], mode="scoring")
        if not report["valid"]:
            raise DomainError("ROW_INVALID", "Raw features failed validation.", report["errors"])
        selected = normalize_records(selected)
        for column in names:
            if column in CATEGORIES:
                if not selected[column].isin(CATEGORIES[column]).all():
                    raise DomainError("ROW_INVALID", "Context category is invalid.", {"column": column})
            else:
                if selected[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
                    raise DomainError("ROW_INVALID", "Boolean is not a numeric feature.", {"column": column})
                selected[column] = pd.to_numeric(selected[column], errors="coerce")
                if not np.isfinite(selected[column].to_numpy(dtype=float)).all():
                    raise DomainError("ROW_INVALID", "Numeric feature must be finite.", {"column": column})
        return selected

    def fit(self, X, y=None):
        self._select(X)
        self.n_features_in_ = len(X.columns)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.raw_feature_names_ = np.asarray(raw_features(self.feature_set), dtype=object)
        self.output_feature_names_ = np.asarray(engineered_features(self.feature_set), dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "output_feature_names_")
        selected = self._select(X)
        selected["previously_contacted"] = (selected["pdays"] != 999).astype(np.int64)
        selected["pdays_value"] = selected["pdays"].where(selected["pdays"] != 999, 0)
        return selected.loc[:, self.output_feature_names_].copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_feature_names_")
        return self.output_feature_names_.copy()

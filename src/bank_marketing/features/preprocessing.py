"""Train-fit-only dense preprocessing shared by all model families."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_marketing.features.engineering import FeatureEngineer, engineered_features
from bank_marketing.schemas import CATEGORIES


def build_pipeline(estimator, feature_set="core_v1", scale=False):
    names = engineered_features(feature_set)
    numeric = [name for name in names if name not in CATEGORIES]
    categorical = [name for name in names if name in CATEGORIES]
    preprocessor = ColumnTransformer(
        [("numeric", StandardScaler() if scale else "passthrough", numeric),
         ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64), categorical)],
        remainder="drop", sparse_threshold=0,
    )
    return Pipeline([("features", FeatureEngineer(feature_set)),
                     ("preprocessor", preprocessor), ("estimator", estimator)])

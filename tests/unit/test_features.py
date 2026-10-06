"""Real estimator and fitted preprocessing paths on synthetic records."""

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from bank_marketing.data.ingest import read_csv
from bank_marketing.features.engineering import FeatureEngineer
from bank_marketing.features.preprocessing import build_pipeline
from bank_marketing.schemas import CORE_FEATURES, CONTEXT_FEATURES, DomainError


@pytest.fixture
def records():
    root = Path(__file__).resolve().parents[2]
    data = read_csv(root / "tests" / "fixtures" / "demo_valid.csv").drop(columns="record_id")
    return data.iloc[np.arange(40) % 4].reset_index(drop=True)


def test_pdays_sentinel_and_stable_order(records):
    records.loc[0, "pdays"] = 999
    records.loc[1, "pdays"] = 0
    features = FeatureEngineer().fit_transform(records.loc[:, list(reversed(CORE_FEATURES))])
    assert features["previously_contacted"].iloc[:2].tolist() == [0, 1]
    assert features["pdays_value"].iloc[:2].tolist() == [0, 0]
    assert "pdays" not in features and len(features.columns) == 11
    engineer = FeatureEngineer().fit(records)
    assert list(engineer.get_feature_names_out()) == list(features.columns)
    assert list(clone(engineer).fit_transform(records).columns) == list(features.columns)


def test_forbidden_inputs_never_enter_model(records):
    extra = records.assign(duration=99999, campaign=99999, y="yes", row_id=np.arange(len(records)), phone="private")
    engineer = FeatureEngineer()
    output = engineer.fit_transform(extra)
    pd.testing.assert_frame_equal(output, FeatureEngineer().fit_transform(records))
    pipeline = build_pipeline(DecisionTreeClassifier(random_state=42))
    pipeline.fit(extra, np.arange(len(records)) % 2)
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    assert not any(any(forbidden in name for forbidden in ("duration", "campaign", "row_id", "phone")) for name in names)
    assert np.array_equal(pipeline.predict(records), pipeline.predict(extra))


def test_encoder_scaler_fit_train_only_and_unseen_safe(records):
    train = records.iloc[:20].copy()
    train["job"] = "admin."
    train["age"] = np.arange(20) + 20
    validation = records.iloc[20:].copy()
    validation["job"] = "student"
    validation["age"] = 100
    pipeline = build_pipeline(LogisticRegression(max_iter=2000), scale=True)
    pipeline.fit(train, np.arange(len(train)) % 2)
    preprocessor = pipeline.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["categorical"]
    scaler = preprocessor.named_transformers_["numeric"]
    assert encoder.categories_[0].tolist() == ["admin."]
    assert scaler.mean_[0] == 29.5
    before_mean = scaler.mean_.copy()
    transformed = pipeline[:-1].transform(validation)
    assert isinstance(transformed, np.ndarray) and transformed.dtype == np.float64
    job_columns = [index for index, name in enumerate(preprocessor.get_feature_names_out()) if "job_" in name]
    assert np.all(transformed[:, job_columns] == 0)
    assert np.isfinite(pipeline.predict_proba(validation)).all()
    np.testing.assert_array_equal(scaler.mean_, before_mean)
    assert encoder.categories_[0].tolist() == ["admin."]


@pytest.mark.parametrize("estimator,scale", [(LogisticRegression(max_iter=2000), True), (GaussianNB(), True),
    (DecisionTreeClassifier(random_state=42), False), (RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=1), False),
    (GradientBoostingClassifier(n_estimators=5, random_state=42), False)])
def test_all_five_families_real_pipeline_and_serialization(records, estimator, scale):
    pipeline = build_pipeline(estimator, scale=scale)
    pipeline.fit(records, np.arange(len(records)) % 2)
    assert list(pipeline.named_steps) == ["features", "preprocessor", "estimator"]
    probabilities = pipeline.predict_proba(records)
    assert probabilities.shape == (40, 2) and np.isfinite(probabilities).all()
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_transformer = preprocessor.named_transformers_["numeric"]
    if scale:
        assert hasattr(numeric_transformer, "mean_")
    else:
        engineered = pipeline.named_steps["features"].transform(records)
        numeric_columns = preprocessor.transformers_[0][2]
        np.testing.assert_allclose(numeric_transformer.transform(engineered[numeric_columns]), engineered[numeric_columns])
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    buffer.seek(0)
    np.testing.assert_allclose(joblib.load(buffer).predict_proba(records), probabilities, atol=1e-8)
    assert len(pipeline[:-1].get_feature_names_out()) == pipeline[:-1].transform(records).shape[1]


def test_context_features_explicit_and_numeric(records):
    context = records.assign(contact="cellular", month="jan", day_of_week="mon")
    for column in ("emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"):
        context[column] = -1.0
    engineer = FeatureEngineer("context_v1").fit(context)
    assert len(engineer.get_feature_names_out()) == 19
    assert list(engineer.raw_feature_names_) == list(CONTEXT_FEATURES)
    assert "contact" not in FeatureEngineer().fit_transform(context)
    pipeline = build_pipeline(DecisionTreeClassifier(random_state=42), "context_v1")
    pipeline.fit(context, np.arange(len(context)) % 2)
    assert np.isfinite(pipeline.predict_proba(context)).all()
    context.loc[0, "month"] = "JAN"
    with pytest.raises(DomainError):
        engineer.transform(context)


def test_feature_validation(records):
    with pytest.raises(DomainError):
        FeatureEngineer("invalid").fit(records)
    with pytest.raises(DomainError):
        FeatureEngineer().fit(records.drop(columns="age"))
    records.loc[0, "job"] = "typo"
    with pytest.raises(DomainError):
        FeatureEngineer().fit(records)

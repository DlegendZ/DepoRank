import csv
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression

from bank_marketing.features.preprocessing import build_pipeline
from bank_marketing.ml.registry import ModelRef
from bank_marketing.schemas import CORE_FEATURES, DomainError
from bank_marketing.services.export import export_errors, export_results
from bank_marketing.services.ranking import rank_records
from bank_marketing.services.scoring import positive_probabilities, score_records


@pytest.fixture
def records():
    return pd.DataFrame([
        {"record_id": f"synthetic-{i}", "age": 25 + i % 45, "job": "admin." if i % 2 else "technician",
         "marital": "married" if i % 3 else "single", "education": "high.school",
         "default": "unknown" if i % 3 else "no", "housing": "yes" if i % 2 else "no", "loan": "no",
         "pdays": 999 if i % 2 else 10, "previous": 0 if i % 2 else 1,
         "poutcome": "nonexistent" if i % 2 else "failure"}
        for i in range(60)
    ])


@pytest.fixture
def model_ref(records):
    pipeline = build_pipeline(LogisticRegression(max_iter=1000), feature_set="core_v1", scale=True)
    x = records.loc[:, list(CORE_FEATURES)]
    y = np.array([int(i % 3 == 0) for i in range(len(x))])
    pipeline.fit(x.iloc[:40], y[:40])
    frozen = FrozenEstimator(pipeline)
    calibrated = CalibratedClassifierCV(frozen, method="sigmoid").fit(x.iloc[40:], y[40:])
    categories = {name: sorted(x.iloc[:40][name].unique().tolist()) for name in x.select_dtypes(include="object").columns}
    return ModelRef("synthetic-run", "logistic_regression", "core_v1", 0.5, calibrated,
                    {"raw_feature_names": list(CORE_FEATURES), "training_categories": categories}, "synthetic-hash")


def test_single_and_batch_probability_mapping_and_provenance(records, model_ref):
    batch = records.iloc[[5, 2, 9]].copy()
    batch.index = [100, 50, 7]
    result = score_records(batch, model_ref)
    expected = positive_probabilities(model_ref.model, batch.loc[:, list(CORE_FEATURES)])
    np.testing.assert_allclose([row["p_yes"] for row in result["rows"]], expected)
    assert [row["input_row_id"] for row in result["rows"]] == [0, 1, 2]
    assert [row["record_id"] for row in result["rows"]] == batch["record_id"].tolist()
    assert result["run_id"] == model_ref.run_id and result["model_hash"] == model_ref.model_hash
    assert result["created_at_utc"]
    single = score_records(batch.iloc[:1], model_ref)
    assert single["batch_id"] != result["batch_id"]
    assert single["rows"][0]["p_yes"] == pytest.approx(expected[0])
    assert all(row["predicted_label"] == ("yes" if row["p_yes"] >= 0.5 else "no") for row in result["rows"])


def test_generated_ids_do_not_claim_customer_identity(records, model_ref):
    result = score_records(records.iloc[:2].drop(columns="record_id"), model_ref)
    assert [row["record_id"] for row in result["rows"]] == [f"{result['batch_id']}:0", f"{result['batch_id']}:1"]


@pytest.mark.parametrize("column,value", [("age", 0), ("age", 35.5), ("job", "not-a-job"), ("previous", -1), ("pdays", 1000), ("housing", ""), ("loan", None)])
def test_one_invalid_row_rejects_entire_batch(records, model_ref, column, value):
    batch = records.iloc[:3].copy()
    batch[column] = batch[column].astype(object)
    batch.loc[1, column] = value
    with pytest.raises(DomainError) as error:
        score_records(batch, model_ref)
    assert error.value.code == "ROW_INVALID"
    errors = error.value.details["errors"]
    assert any(item["input_row_id"] == 1 and item["column"] == column for item in errors)
    exported = list(csv.DictReader(io.StringIO(export_errors(errors).decode("utf-8-sig"))))
    assert any(row["input_row_id"] == "1" for row in exported)


def test_ordinary_labels_rejected_and_demo_explicitly_discards(records, model_ref):
    batch = records.iloc[:3].copy()
    batch["y"] = ["yes", "no", "yes"]
    batch["duration"] = [5, 20, 120]
    batch["campaign"] = 2
    for column, value in {"contact": "cellular", "month": "may", "day_of_week": "mon",
                          "emp.var.rate": 1.1, "cons.price.idx": 93.0, "cons.conf.idx": -42.0,
                          "euribor3m": 4.0, "nr.employed": 5000.0}.items():
        batch[column] = value
    before = batch.copy(deep=True)
    with pytest.raises(DomainError) as error:
        score_records(batch, model_ref)
    assert error.value.code == "SCHEMA_INVALID"
    demo = score_records(batch, model_ref, mode="demo")
    plain = score_records(batch.loc[:, ["record_id", *CORE_FEATURES]], model_ref)
    np.testing.assert_allclose([row["p_yes"] for row in demo["rows"]], [row["p_yes"] for row in plain["rows"]])
    discarded = {warning["column"] for warning in demo["warnings"] if warning.get("code") == "DEMO_COLUMN_DISCARDED"}
    assert {"y", "duration", "campaign"} <= discarded
    assert not any(key in demo for key in ["metrics", "performance", "y"])
    assert all("y" not in row for row in demo["rows"])
    pd.testing.assert_frame_equal(batch, before)


def test_legitimate_unseen_categories_warn_and_illegal_categories_reject(records, model_ref):
    batch = records.iloc[:2].copy()
    batch.loc[0, "job"] = "student"
    batch.loc[1, "education"] = "illiterate"
    result = score_records(batch, model_ref)
    assert any(warning["code"] == "UNSEEN_CATEGORY" and warning["column"] == "job" and warning["input_row_id"] == 0 for warning in result["rows"][0]["warnings"])
    assert any(warning["code"] == "UNSEEN_CATEGORY" and warning["column"] == "education" for warning in result["rows"][1]["warnings"])
    assert all(np.isfinite(row["p_yes"]) for row in result["rows"])
    batch.loc[0, "job"] = "STUDENT"
    with pytest.raises(DomainError):
        score_records(batch, model_ref)


def test_whitespace_normalization_semantic_warnings_and_duplicate_record_ids(records, model_ref):
    batch = records.iloc[:2].copy()
    batch.loc[0, "job"] = " admin. "
    batch.loc[0, "pdays"] = 999
    batch.loc[0, "previous"] = 1
    result = score_records(batch, model_ref)
    assert result["warnings"]
    assert batch.loc[0, "job"] == " admin. "
    batch["record_id"] = "same"
    with pytest.raises(DomainError):
        score_records(batch, model_ref)


def test_scoring_ranking_export_and_serialization_parity(records, model_ref, tmp_path):
    artifact = tmp_path / "calibrated.joblib"
    joblib.dump(model_ref.model, artifact)
    before = positive_probabilities(model_ref.model, records.loc[:, list(CORE_FEATURES)])
    loaded = joblib.load(artifact)
    np.testing.assert_allclose(positive_probabilities(loaded, records.loc[:, list(CORE_FEATURES)]), before, atol=1e-8)
    model_ref.model = loaded
    batch = records.iloc[:5].copy()
    batch.loc[0, "record_id"] = " \t=SUM(1,2)"
    ranked = rank_records(score_records(batch, model_ref), {"q": 0.1})
    assert ranked["capacity_k"] == 1
    csv_rows = list(csv.DictReader(io.StringIO(export_results(ranked, include_features=batch.drop(columns="record_id")).decode("utf-8-sig"))))
    assert len(csv_rows) == len(batch)
    assert sum(row["selected_for_contact"] == "True" for row in csv_rows) == 1
    assert [int(row["input_row_id"]) for row in csv_rows] == [row["input_row_id"] for row in ranked["rows"]]
    assert next(row for row in csv_rows if row["input_row_id"] == "0")["record_id"].startswith("'")
    assert len(json.loads(export_results(ranked, format="json", top_k_only=True))["rows"]) == 1


def test_no_registry_ready_and_corrupt_metadata(records, model_ref):
    with pytest.raises(DomainError) as error:
        score_records(records, None)
    assert error.value.code == "MODEL_NOT_READY"
    model_ref.threshold = float("nan")
    with pytest.raises(DomainError) as error:
        score_records(records, model_ref)
    assert error.value.code == "ARTIFACT_INTEGRITY_ERROR"
    model_ref.threshold = 0.5
    model_ref.metadata["raw_feature_names"] = ["duration"]
    with pytest.raises(DomainError) as error:
        score_records(records, model_ref)
    assert error.value.code == "ARTIFACT_INTEGRITY_ERROR"


def test_duplicate_columns_empty_batch_and_forbidden_identifiers(records, model_ref):
    with pytest.raises(DomainError):
        score_records(records.iloc[:0], model_ref)
    duplicate = pd.concat([records.iloc[:2], records.iloc[:2][["job"]]], axis=1)
    with pytest.raises(DomainError):
        score_records(duplicate, model_ref)
    forbidden = records.iloc[:2].assign(input_row_id=[4, 5])
    with pytest.raises(DomainError):
        score_records(forbidden, model_ref)
def test_threshold_equality_is_positive_and_capacity_independent(records, model_ref):
    batch = records.iloc[:1]
    probability = positive_probabilities(model_ref.model, batch.loc[:, list(CORE_FEATURES)])[0]
    model_ref.threshold = float(probability)
    result = score_records(batch, model_ref)
    assert result["rows"][0]["predicted_label"] == "yes"
    assert not rank_records(result, {"k": 0})["rows"][0]["selected_for_contact"]
    model_ref.threshold = float(np.nextafter(probability, 1.0))
    result = score_records(batch, model_ref)
    assert result["rows"][0]["predicted_label"] == "no"
    assert rank_records(result, {"k": 1})["rows"][0]["selected_for_contact"]
def test_invalid_batch_rejected_before_unfitted_real_model_inference(records, model_ref):
    model_ref.model = build_pipeline(LogisticRegression(), feature_set="core_v1", scale=True)
    batch = records.iloc[:2].copy()
    batch.loc[1, "age"] = 0
    with pytest.raises(DomainError) as error:
        score_records(batch, model_ref)
    assert error.value.code == "ROW_INVALID"
    assert error.value.details["errors"][0]["input_row_id"] == 1
def test_core_labeled_demo_and_raw_feature_order(records, model_ref):
    batch = records.iloc[:2].assign(y=["yes", "no"])
    model_ref.metadata["raw_feature_names"] = list(reversed(CORE_FEATURES))
    result = score_records(batch, model_ref, mode="demo")
    expected = positive_probabilities(model_ref.model, batch.loc[:, list(CORE_FEATURES)])
    np.testing.assert_allclose([row["p_yes"] for row in result["rows"]], expected)
    assert [warning["column"] for warning in result["warnings"] if warning["code"] == "DEMO_COLUMN_DISCARDED"] == ["y"]

@pytest.mark.parametrize('suffix', ['csv', 'json'])
def test_cli_score_format_and_metadata(tmp_path, suffix):
    import json
    from bank_marketing.cli import main
    if not Path('artifacts/active_run.json').exists():
        pytest.skip('Trusted active run required')
    output = tmp_path / f'priorities.{suffix}'
    assert main(['score', '--input', 'tests/fixtures/demo_valid.csv', '--k', '1', '--output', str(output)]) == 0
    if suffix == 'json':
        assert len(json.loads(output.read_text())['rows']) == 4
    else:
        assert len(pd.read_csv(output)) == 4
    meta = json.loads(output.with_suffix('.metadata.json').read_text())
    assert meta['batch_size'] == 4
    assert meta['policy']['invalid_policy'] == 'reject_batch'
def test_cli_score_cannot_overwrite_input(tmp_path):
    from bank_marketing.cli import main
    source = tmp_path / 'input.csv'
    content = Path('tests/fixtures/demo_valid.csv').read_bytes()
    source.write_bytes(content)
    assert main(['score', '--input', str(source), '--output', str(source)]) == 2
    assert source.read_bytes() == content
def test_cli_score_rejects_non_export_extension(tmp_path):
    from bank_marketing.cli import main
    output = tmp_path / 'result.joblib'
    assert main(['score', '--input', 'tests/fixtures/demo_valid.csv', '--output', str(output)]) == 2
    assert not output.exists()
def test_cli_score_rejects_raw_destination():
    from bank_marketing.cli import main
    destination = Path('data/raw/quality_probe.csv')
    assert main(['score', '--input', 'tests/fixtures/demo_valid.csv', '--output', str(destination)]) == 2
    assert not destination.exists()
@pytest.mark.parametrize('target', ['metadata', 'errors', 'hardlink'])
def test_cli_score_rejects_output_collisions(tmp_path, target):
    import os
    from bank_marketing.cli import main
    source = tmp_path / 'input.csv'
    source.write_bytes(Path('tests/fixtures/demo_valid.csv').read_bytes())
    output = tmp_path / 'priorities.csv'
    arguments = ['score', '--input', str(source), '--output', str(output)]
    if target == 'metadata':
        output = tmp_path / 'input.metadata.csv'
        arguments[-1] = str(output)
        source = output.with_suffix('.metadata.json')
        source.write_bytes(Path('tests/fixtures/demo_valid.csv').read_bytes())
        arguments[2] = str(source)
    elif target == 'errors':
        arguments += ['--errors-output', str(output)]
    else:
        os.link(source, output)
    before = source.read_bytes()
    assert main(arguments) == 2
    assert source.read_bytes() == before

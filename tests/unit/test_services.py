import csv
import io
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from bank_marketing.schemas import DomainError
from bank_marketing.services.export import ERROR_COLUMNS, RESULT_COLUMNS, export_errors, export_metadata, export_results
from bank_marketing.services.ranking import rank_records
from bank_marketing.services.scoring import positive_probabilities


@pytest.fixture
def scores():
    return {
        "batch_id": "batch", "run_id": "run", "model_id": "M01", "feature_set": "core_v1",
        "threshold": 0.5, "model_hash": "abc", "warnings": [],
        "rows": [
            {"input_row_id": i, "record_id": f"id-{i}", "p_yes": p, "predicted_label": "yes" if p >= 0.5 else "no", "warnings": []}
            for i, p in [(2, 0.4), (1, 0.8), (3, 0.1), (0, 0.8)]
        ],
    }


def read_csv(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def test_hand_computed_fixture_stable_ties_and_copy(scores):
    original = deepcopy(scores)
    ranked = rank_records(scores, {"k": 2})
    assert scores == original
    assert [row["input_row_id"] for row in ranked["rows"]] == [0, 1, 2, 3]
    assert [row["rank"] for row in ranked["rows"]] == [1, 2, 3, 4]
    assert [row["selected_for_contact"] for row in ranked["rows"]] == [True, True, False, False]
    y = {0: 1, 1: 0, 2: 1, 3: 0}
    hits = sum(y[row["input_row_id"]] for row in ranked["rows"] if row["selected_for_contact"])
    assert hits / 2 == 0.5
    assert hits / sum(y.values()) == 0.5
    assert (hits / 2) / (sum(y.values()) / len(y)) == 1.0
    assert ranked["capacity_k"] == 2 and ranked["call_fraction"] == 0.5
    ranked["rows"][0]["warnings"].append("changed")
    assert scores == original


@pytest.mark.parametrize("capacity,k", [({"k": 0}, 0), ({"k": 4}, 4), ({"q": 0}, 0), ({"q": 1}, 4), ({"q": 0.1}, 1), ({"q": 0.51}, 3), ({"k": np.int64(2)}, 2)])
def test_capacity_valid_edges(scores, capacity, k):
    result = rank_records(scores, capacity)
    assert result["capacity_k"] == k
    assert sum(row["selected_for_contact"] for row in result["rows"]) == k


@pytest.mark.parametrize("capacity", [{}, {"k": 1, "q": 0.5}, {"k": -1}, {"k": 5}, {"k": 1.0}, {"k": True}, {"k": "2"}, {"k": None}, {"q": -0.1}, {"q": 1.1}, {"q": float("nan")}, {"q": float("inf")}, {"q": True}, {"q": "0.1"}, {"q": None}, {"x": 2}, None])
def test_capacity_invalid_no_clipping(scores, capacity):
    with pytest.raises(DomainError) as error:
        rank_records(scores, capacity)
    assert error.value.code == "CAPACITY_INVALID"


def test_empty_batch_capacity_and_labels_independent(scores):
    assert rank_records({"rows": []}, {"q": 1})["call_fraction"] == 0
    result = rank_records(scores, {"k": 3})
    assert result["rows"][2]["predicted_label"] == "no"
    assert result["rows"][2]["selected_for_contact"] is True
    assert rank_records(scores, {"k": 0})["rows"][0]["predicted_label"] == "yes"


@pytest.mark.parametrize("field,value", [("p_yes", float("nan")), ("p_yes", -0.1), ("p_yes", 1.1), ("p_yes", True), ("input_row_id", -1), ("input_row_id", True), ("input_row_id", 1.0)])
def test_invalid_score_row(scores, field, value):
    scores["rows"][0][field] = value
    with pytest.raises(DomainError):
        rank_records(scores, {"k": 1})


def test_duplicate_input_row_ids(scores):
    scores["rows"][0]["input_row_id"] = 1
    with pytest.raises(DomainError):
        rank_records(scores, {"k": 1})


@pytest.mark.parametrize("text", ["=SUM(A1)", "+cmd", "-cmd", "@SUM(A1)", " \t=1", "\r\n+1", "\x00@X", "\ufeff=1", "\u200b-1"])
def test_csv_formula_strings_sanitized(scores, text):
    scores["rows"][0]["record_id"] = text
    row = read_csv(export_results(scores))[0]
    assert row["record_id"] == "'" + text
    assert json.loads(export_results(scores, format="json"))["rows"][0]["record_id"] == text


def test_export_precision_all_rows_top_k_and_feature_mapping(scores):
    ranked = rank_records(scores, {"k": 2})
    ranked["rows"][0]["p_yes"] = 0.8123456789012345
    features = pd.DataFrame({"age": [31, 42, 53, 64], "cons.conf.idx": [-42.7, -33.2, -22.1, -11.0]})
    rows = read_csv(export_results(ranked, include_features=features))
    assert list(rows[0]) == [*RESULT_COLUMNS, "age", "cons.conf.idx"]
    assert len(rows) == 4
    assert float(rows[0]["p_yes"]) == ranked["rows"][0]["p_yes"]
    assert rows[0]["age"] == "31" and rows[0]["cons.conf.idx"] == "-42.7"
    assert len(read_csv(export_results(ranked, top_k_only=True))) == 2
    payload = json.loads(export_results(ranked, format="json", top_k_only=True, include_features=features))
    assert len(payload["rows"]) == 2 and payload["rows"][0]["age"] == 31
    assert len(ranked["rows"]) == 4
    metadata = json.loads(export_metadata(ranked))
    assert metadata["model_hash"] == "abc" and metadata["batch_size"] == 4


def test_features_attach_by_input_id_not_rank(scores):
    ranked = rank_records(scores, {"k": 4})
    features = pd.DataFrame({"input_row_id": [3, 2, 1, 0], "age": [64, 53, 42, 31]})
    assert [row["age"] for row in read_csv(export_results(ranked, include_features=features))] == ["31", "42", "53", "64"]


@pytest.mark.parametrize("format", ["csv", "json"])
def test_exports_reject_nonfinite(scores, format):
    scores["rows"][0]["p_yes"] = float("nan")
    with pytest.raises(DomainError):
        export_results(scores, format=format)


def test_strict_json_metadata_and_generated_column_collision(scores):
    scores["threshold"] = float("inf")
    with pytest.raises(DomainError):
        export_results(scores, format="json")
    with pytest.raises(DomainError):
        export_results(scores, include_features=pd.DataFrame({"p_yes": [0.0] * 4}))
    with pytest.raises(DomainError):
        export_results(scores, format="xml")


def test_error_report_no_raw_data_or_traceback():
    data = [{"input_row_id": 2, "column": "job", "code": "ROW_INVALID", "message": " \t=invalid", "value": "PRIVATE", "traceback": "PRIVATE"}]
    rows = read_csv(export_errors(data))
    assert list(rows[0]) == list(ERROR_COLUMNS)
    assert rows[0]["input_row_id"] == "2" and rows[0]["message"] == "' \t=invalid"
    assert "PRIVATE" not in export_errors(data).decode("utf-8-sig")


def test_positive_class_order_and_missing_class_real_models():
    x = pd.DataFrame({"x": [0, 1, 2, 3]})
    model = DummyClassifier(strategy="prior").fit(x, [0, 0, 0, 1])
    np.testing.assert_allclose(positive_probabilities(model, x), 0.25)
    model.classes_ = model.classes_[::-1]
    np.testing.assert_allclose(positive_probabilities(model, x), model.predict_proba(x)[:, 0])
    missing = DummyClassifier(strategy="prior").fit(x, [0, 0, 0, 0])
    with pytest.raises(DomainError) as error:
        positive_probabilities(missing, x)
    assert error.value.code == "ARTIFACT_INTEGRITY_ERROR"
    with pytest.raises(DomainError) as error:
        positive_probabilities(None, x)
    assert error.value.code == "MODEL_NOT_READY"
def test_json_features_require_opt_in(scores):
    scores["rows"][0]["age"] = 42
    default = json.loads(export_results(scores, format="json"))
    assert "age" not in default["rows"][0]
    opted = json.loads(export_results(scores, format="json", include_features=["age"]))
    assert opted["rows"][0]["age"] == 42
@pytest.mark.parametrize("bad_prior", [[float("nan"), float("nan")], [-0.1, 1.1]])
def test_probability_integrity_rejects_corrupt_real_fitted_model(bad_prior):
    x = pd.DataFrame({"x": [0, 1, 2, 3]})
    model = DummyClassifier(strategy="prior").fit(x, [0, 0, 0, 1])
    model.class_prior_ = np.array(bad_prior)
    with pytest.raises(DomainError) as error:
        positive_probabilities(model, x)
    assert error.value.code == "ARTIFACT_INTEGRITY_ERROR"

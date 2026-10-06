import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score

from bank_marketing.ml.evaluate import capacity_metrics, evaluate_scores, positive_probabilities, threshold_for_f1
from bank_marketing.ml.models import refit_index
from bank_marketing.ml.selection import choose_winner
from bank_marketing.schemas import DomainError
from bank_marketing.storage import canonical_bytes, object_hash


def test_manual_srd_metrics():
    report = evaluate_scores([1, 0, 1, 0], [0.8, 0.8, 0.4, 0.1], 0.5)
    assert report["confusion_matrix"] == [[1, 1], [1, 1]]
    metrics = report["metrics"]
    assert metrics["f1"] == 0.5
    assert metrics["average_precision"] == pytest.approx(7 / 12)
    assert metrics["brier_score"] == pytest.approx(0.2625)
    cap = capacity_metrics([1, 0, 1, 0], [0.8, 0.8, 0.4, 0.1], 2)
    assert cap["precision_at_k"] == cap["recall_at_k"] == 0.5
    assert cap["lift_at_k"] == 1.0


def test_zero_denominators_are_null_and_json_safe():
    cap = capacity_metrics([0, 0], [0.2, 0.3], 0)
    assert cap["precision_at_k"] is None
    result = evaluate_scores([0, 0], [0.2, 0.3], 1)
    assert result["metrics"]["roc_auc"] is None
    assert result["metrics"]["average_precision"] is None
    assert result["metrics"]["recall"] is None
    assert b"NaN" not in canonical_bytes(result)
    assert b"Infinity" not in canonical_bytes(result)


def test_prior_ap_equals_evaluation_prevalence():
    model = DummyClassifier(strategy="prior").fit(np.zeros((4, 1)), [0, 0, 0, 1])
    p = positive_probabilities(model, np.zeros((6, 1)))
    assert np.all(p == 0.25)
    assert average_precision_score([0, 1, 1, 0, 1, 0], p) == 0.5


def test_threshold_ties_choose_highest():
    assert threshold_for_f1([1, 0], [0.8, 0.2]) == 0.8


def test_inner_refit_std_then_serialized_params():
    cv = {"params": [{"C": 2}, {"C": 1}, {"C": 0.1}], "mean_test_score": [0.5, 0.5, 0.4], "std_test_score": [0.1, 0.1, 0.01]}
    assert refit_index(cv) == 1
    cv["std_test_score"][0] = 0.09
    assert refit_index(cv) == 0


def test_winner_practical_tie_and_all_tiebreaks():
    rows = [dict(model_id="a", average_precision=0.50, precision_at_k=0.4, brier_score=0.1, inference_latency=0.01),
            dict(model_id="b", average_precision=0.496, precision_at_k=0.5, brier_score=0.2, inference_latency=0.02),
            dict(model_id="c", average_precision=0.49, precision_at_k=0.9, brier_score=0.01, inference_latency=0.001)]
    assert choose_winner(rows)["winner_model_id"] == "b"
    rows[0]["precision_at_k"] = 0.5
    assert choose_winner(rows)["winner_model_id"] == "a"
    rows[1]["brier_score"] = 0.1
    rows[1]["inference_latency"] = 0.005
    assert choose_winner(rows)["winner_model_id"] == "b"
    rows[1]["inference_latency"] = 0.01
    assert choose_winner(rows)["winner_model_id"] == "a"


def test_canonical_hash_order_and_nonfinite_guard():
    assert object_hash({"b": 2, "a": 1}) == object_hash({"a": 1, "b": 2})
    with pytest.raises(ValueError):
        canonical_bytes({"p": float("nan")})


def test_cap_no_clipping():
    with pytest.raises(DomainError):
        capacity_metrics([0, 1], [0.1, 0.9], 3)


@pytest.mark.parametrize("labels,scores", [
    ([0.1, 1.9], [0.2, 0.8]), ([0, 2], [0.2, 0.8]),
    ([[0], [1]], [0.2, 0.8]), ([0, 1], [[0.2], [0.8]]),
    ([0, 1], [True, False]), ([0, 1], [0.2]), ([], []),
])
def test_metric_inputs_rejected_without_coercion(labels, scores):
    for evaluate in (evaluate_scores, threshold_for_f1):
        with pytest.raises(DomainError):
            evaluate(labels, scores)
    with pytest.raises(DomainError):
        capacity_metrics(labels, scores, 0)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1, True, "0.5"])
def test_metric_threshold_strict(threshold):
    with pytest.raises(DomainError):
        evaluate_scores([0, 1], [0.2, 0.8], threshold)


@pytest.mark.parametrize("q", [float("nan"), float("inf"), -0.1, 1.1, True, "0.1"])
def test_metric_fraction_strict(q):
    with pytest.raises(DomainError):
        evaluate_scores([0, 1], [0.2, 0.8], q=q)


@pytest.mark.parametrize("ids", [[0], [0, 0], [-1, 1], [0.5, 1], [False, True], [[0], [1]]])
def test_metric_row_ids_strict(ids):
    with pytest.raises(DomainError):
        evaluate_scores([0, 1], [0.2, 0.8], row_ids=ids)


def test_evaluation_exact_k_without_fraction_roundtrip():
    result = evaluate_scores([0, 1] * 25, [0.2, 0.8] * 25, k=7)
    assert result["metrics"]["capacity_k"] == 7
    assert result["metrics"]["precision_at_k"] == 1
    assert evaluate_scores([0, 1], [0.2, 0.8], k=0)["metrics"]["precision_at_k"] is None

def test_ml_probability_shape_and_class_order_guard():
    X = np.zeros((4, 1))
    model = DummyClassifier(strategy='prior').fit(X, [0, 0, 0, 1])
    model.class_prior_ = np.array([0.2, 0.3, 0.5])
    with pytest.raises(DomainError) as failure:
        positive_probabilities(model, X)
    assert failure.value.code == 'ARTIFACT_INTEGRITY_ERROR'
    with pytest.raises(DomainError):
        positive_probabilities(None, X)
def test_evaluation_rejects_ambiguous_capacity_modes():
    with pytest.raises(DomainError) as failure:
        evaluate_scores([0, 1], [0.2, 0.8], k=1, q=0.9)
    assert failure.value.code == 'CAPACITY_INVALID'

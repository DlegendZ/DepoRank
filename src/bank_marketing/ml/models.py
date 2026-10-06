import json

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from bank_marketing.schemas import DomainError

MODEL_CLASSES = {"logistic_regression": LogisticRegression, "gaussian_nb": GaussianNB,
                 "decision_tree": DecisionTreeClassifier, "random_forest": RandomForestClassifier,
                 "gradient_boosting": GradientBoostingClassifier}


def estimator_for(model_id, config):
    if model_id not in MODEL_CLASSES:
        raise DomainError("SCHEMA_INVALID", "Model tidak dikenal.")
    return MODEL_CLASSES[model_id](**config["model_parameters"][model_id])


def refit_index(results):
    return min(range(len(results["params"])), key=lambda i: (
        -float(results["mean_test_score"][i]), float(results["std_test_score"][i]),
        json.dumps(results["params"][i], sort_keys=True, separators=(",", ":"))))


def assert_binary(y):
    if not np.array_equal(np.unique(y), [0, 1]):
        raise DomainError("TRAINING_FAILED", "Fit/evaluasi membutuhkan kedua kelas.")

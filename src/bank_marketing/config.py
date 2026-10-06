"""Validate the academic baseline and resolve all estimator parameters."""

from copy import deepcopy
from pathlib import Path

import yaml

from bank_marketing.schemas import DATASET_SHA256, SPLIT_RANGES, DomainError

MODEL_IDS = ("logistic_regression", "gaussian_nb", "decision_tree", "random_forest", "gradient_boosting")
GRIDS = {
    "logistic_regression": {"estimator__C": [0.1, 1, 10], "estimator__class_weight": [None, "balanced"]},
    "gaussian_nb": {"estimator__var_smoothing": [1e-11, 1e-9, 1e-7]},
    "decision_tree": {"estimator__max_depth": [3, 6, 10, None], "estimator__min_samples_leaf": [1, 20], "estimator__class_weight": [None, "balanced"]},
    "random_forest": {"estimator__max_depth": [8, None], "estimator__min_samples_leaf": [1, 10], "estimator__class_weight": [None, "balanced"]},
    "gradient_boosting": {"estimator__n_estimators": [100, 200], "estimator__learning_rate": [0.05, 0.1], "estimator__max_depth": [2, 3]},
}
MODEL_PARAMETERS = {
    "logistic_regression": {"solver": "lbfgs", "max_iter": 2000},
    "gaussian_nb": {},
    "decision_tree": {"random_state": 42},
    "random_forest": {"n_estimators": 200, "max_features": "sqrt", "random_state": 42, "n_jobs": 2},
    "gradient_boosting": {"random_state": 42},
}
BASELINE = {
    "schema_version": "1.0", "seed": 42,
    "dataset": {"variant": "bank-additional-full", "path": "data/raw/bank-additional-full.csv", "separator": ";", "expected_rows": 41188, "sha256": DATASET_SHA256},
    "features": {"set": "core_v1", "forbidden": ["duration", "campaign", "y", "row_id"]},
    "split": {**SPLIT_RANGES, "bounds": "half_open"},
    "cv": {"kind": "expanding_window", "n_splits": 3, "gap": 0},
    "search": {"scoring": "average_precision", "n_jobs": 1, "error_score": "raise"},
    "models": list(MODEL_IDS),
    "calibration": {"method": "sigmoid", "frozen_base": True},
    "selection": {"primary_metric": "average_precision", "practical_tie_delta": 0.005, "tie_breakers": ["precision_at_fraction", "brier_score", "inference_latency", "model_id"], "threshold_objective": "f1", "threshold_grid_points": 101},
    "capacity": {"default_fraction": 0.10, "extra_evaluation_k": 1000},
    "upload": {"max_bytes": 20000000, "max_rows": 50000, "invalid_policy": "reject_batch"},
}


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise DomainError("SCHEMA_INVALID", "Duplicate configuration key.", {"key": key})
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _invalid(message, field):
    raise DomainError("SCHEMA_INVALID", message, {"field": field})


def validate_config(config):
    if not isinstance(config, dict):
        _invalid("Configuration must be a mapping.", "config")
    for section, default in BASELINE.items():
        if section not in config:
            _invalid("Required configuration field is missing.", section)
        if isinstance(default, dict):
            if not isinstance(config[section], dict):
                _invalid("Configuration section must be a mapping.", section)
            for field in default:
                if field not in config[section]:
                    _invalid("Required configuration field is missing.", f"{section}.{field}")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        _invalid("Seed must be a nonnegative 32-bit integer.", "seed")
    if config["seed"] != 42:
        _invalid("Academic baseline seed is frozen at 42.", "seed")
    if config["schema_version"] != "1.0":
        _invalid("Unsupported schema version.", "schema_version")
    if config["models"] != list(MODEL_IDS):
        _invalid("Academic baseline requires all five known models in SRD order.", "models")
    if config["features"]["set"] not in ("core_v1", "context_v1"):
        _invalid("Unknown feature set.", "features.set")
    if config["features"]["forbidden"] != BASELINE["features"]["forbidden"]:
        _invalid("Forbidden features must match the SRD baseline.", "features.forbidden")
    if not isinstance(config["dataset"]["path"], str) or not config["dataset"]["path"].strip():
        _invalid("Dataset path must be a nonempty string.", "dataset.path")
    for field in ("variant", "separator", "expected_rows", "sha256"):
        if type(config["dataset"][field]) is not type(BASELINE["dataset"][field]) or config["dataset"][field] != BASELINE["dataset"][field]:
            _invalid("Official dataset contract differs from the baseline.", f"dataset.{field}")
    for name, expected in SPLIT_RANGES.items():
        bounds = config["split"][name]
        if not isinstance(bounds, list) or len(bounds) != 2 or any(type(value) is not int for value in bounds) or bounds != expected:
            _invalid("Split must match fixed, contiguous half-open SRD ranges.", f"split.{name}")
    for section in ("cv", "search", "calibration", "selection", "capacity", "upload"):
        for field, expected in BASELINE[section].items():
            value = config[section][field]
            if type(value) is not type(expected) or value != expected:
                _invalid("Academic baseline field is frozen; protocol changes need a new version.", f"{section}.{field}")
    if config["split"]["bounds"] != "half_open":
        _invalid("Split bounds must be half_open.", "split.bounds")
    for section, expected in (("grids", GRIDS), ("model_parameters", MODEL_PARAMETERS)):
        if section in config and config[section] != expected:
            _invalid("All five exact SRD grids and fixed parameters are frozen.", section)
    resolved = deepcopy(config)
    resolved["grids"] = deepcopy(GRIDS)
    resolved["model_parameters"] = deepcopy(MODEL_PARAMETERS)
    return resolved


def load_config(path) -> dict:
    try:
        config = yaml.load(Path(path).read_text(encoding="utf-8-sig"), Loader=_UniqueLoader)
    except (OSError, yaml.YAMLError, TypeError) as exc:
        raise DomainError("SCHEMA_INVALID", "Configuration could not be read as YAML.") from exc
    return validate_config(config)

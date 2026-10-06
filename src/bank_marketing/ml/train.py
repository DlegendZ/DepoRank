from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import time
import uuid
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
import yaml

from bank_marketing.config import validate_config
from bank_marketing.data.audit import audit_dataset
from bank_marketing.data.splits import create_splits, inner_folds
from bank_marketing.features.preprocessing import build_pipeline
from bank_marketing.ml.models import assert_binary, estimator_for, refit_index
from bank_marketing.ml.registry import Registry
from bank_marketing.schemas import CORE_FEATURES, CONTEXT_FEATURES, DomainError
from bank_marketing.storage import (append_log, atomic_bytes, environment, file_hash,
                                    object_hash, read_json, utc_now, write_json)


def save_model(path, model):
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    atomic_bytes(path, buffer.getvalue())


def write_csv(path, frame):
    atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def update_manifest(path, **fields):
    manifest = read_json(path / "run_manifest.json")
    manifest.update(fields)
    write_json(path / "run_manifest.json", manifest)
    append_log(path, {"run_id": manifest["run_id"], "state": manifest["status"]})
    return manifest


def load_run_data(path):
    manifest = read_json(path / "run_manifest.json")
    config = yaml.safe_load((path / "config_resolved.yaml").read_text(encoding="utf-8"))
    if object_hash(config) != manifest["config_sha256"]:
        raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Run configuration has changed.")
    source = Path(config["dataset"]["path"])
    if not source.is_file() or file_hash(source) != manifest["dataset_sha256"]:
        raise DomainError("DATASET_HASH_MISMATCH", "Raw dataset does not match the run.")
    data = pd.read_csv(source, sep=";", encoding="utf-8-sig")
    return data, config


def require_stage(path, status):
    Registry(path.parent.parent).verify_run(path.name, require_lock=False)
    manifest = read_json(path / "run_manifest.json")
    if manifest["status"] != status:
        raise DomainError("MODEL_NOT_READY", f"Tahap membutuhkan status {status}.")
    return manifest


def mark_failure(path, error):
    if (path / "run_manifest.json").exists():
        update_manifest(path, status="failed", finished_at_utc=utc_now(), error_id=getattr(error, "error_id", uuid.uuid4().hex))
        append_log(path, {"code": getattr(error, "code", "TRAINING_FAILED"), "exception_type": type(error).__name__})


def train_models(dataset, config, run_id=None, root="artifacts"):
    config = validate_config(config)
    data = dataset.copy() if isinstance(dataset, pd.DataFrame) else pd.read_csv(dataset, sep=";", encoding="utf-8-sig")
    source = Path(config["dataset"]["path"])
    if not source.exists() or file_hash(source) != config["dataset"]["sha256"]:
        raise DomainError("DATASET_HASH_MISMATCH", "Dataset checksum does not match.")
    report = audit_dataset(data, mode="training")
    if not report["valid"]:
        raise DomainError("SCHEMA_INVALID", "Training audit failed.", report["errors"])
    if len(data) != config["dataset"]["expected_rows"]:
        raise DomainError("DATASET_VARIANT_MISMATCH", "Dataset row count does not match.")
    source_data = pd.read_csv(source, sep=";", encoding="utf-8-sig")
    if not data.reset_index(drop=True).equals(source_data):
        raise DomainError("DATASET_HASH_MISMATCH", "Training must preserve verified raw values and row order.")
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    path = Registry(root).run_path(run_id)
    path.mkdir(parents=True, exist_ok=False)
    env = environment()
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    config = json.loads(json.dumps(config))
    config["dataset"]["path"] = str(source.resolve())
    splits = create_splits(data, config)
    ranges = splits["ranges"]
    raw_features = list(CORE_FEATURES if config["features"]["set"] == "core_v1" else CONTEXT_FEATURES)
    atomic_bytes(path / "config_resolved.yaml", yaml.safe_dump(config, sort_keys=True).encode("utf-8"))
    write_json(path / "environment.json", env)
    write_json(path / "audit.json", report)
    write_json(path / "split_manifest.json", splits)
    ds_manifest = Path("data/manifests/dataset_manifest.json")
    if ds_manifest.exists():
        write_json(path / "dataset_manifest.json", read_json(ds_manifest))
    manifest = {"schema_version": "1.0", "run_id": run_id, "status": "split_ready", "created_at_utc": utc_now(),
                "started_at_utc": utc_now(), "finished_at_utc": None, "dataset_sha256": file_hash(source),
                "config_sha256": object_hash(config), "source_revision": revision.stdout.strip() if revision.returncode == 0 else None,
                **env, "seed": config["seed"], "feature_set": config["features"]["set"],
                "split_manifest_sha256": file_hash(path / "split_manifest.json"), "error_id": None,
                "integrity": {name: file_hash(path / name) for name in ("config_resolved.yaml", "environment.json", "audit.json", "split_manifest.json")}}
    write_json(path / "run_manifest.json", manifest)
    print(f"run_id={run_id}", flush=True)
    train = data.iloc[slice(*ranges["train"])]
    X, y = train[raw_features], train.y.map({"no": 0, "yes": 1}).to_numpy()
    assert_binary(y)
    folds = inner_folds(train, n_splits=config["cv"]["n_splits"], gap=config["cv"]["gap"])
    results = []
    started = time.perf_counter()
    try:
        update_manifest(path, status="training")
        for model_id in config["models"]:
            print(f"Training {model_id}; expanding-window 3 folds", flush=True)
            pipeline = build_pipeline(estimator_for(model_id, config), feature_set=config["features"]["set"], scale=model_id in ("logistic_regression", "gaussian_nb"))
            search = GridSearchCV(pipeline, config["grids"][model_id], scoring="average_precision", cv=folds,
                                  refit=refit_index, error_score="raise", n_jobs=1, verbose=2)
            begin = time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ConvergenceWarning)
                search.fit(X, y)
            if any(issubclass(w.category, ConvergenceWarning) for w in captured):
                raise DomainError("TRAINING_FAILED", "Model did not converge; run stopped.", {"model_id": model_id})
            model_dir = path / "models" / model_id
            save_model(model_dir / "base_pipeline.joblib", search.best_estimator_)
            base = search.best_estimator_
            transformed = list(base.named_steps["preprocessor"].get_feature_names_out())
            metadata = {"schema_version": "1.0", "model_id": model_id,
                        "estimator_class": type(base.named_steps["estimator"]).__name__,
                        "hyperparameters": {**config["model_parameters"][model_id], **search.best_params_},
                        "feature_set": config["features"]["set"], "raw_feature_names": raw_features,
                        "transformed_feature_names": transformed, "transformed_feature_count": len(transformed),
                        "training_categories": {col: sorted(train[col].unique().tolist()) for col in raw_features if train[col].dtype == object},
                        "target_mapping": {"no": 0, "yes": 1}, "probability_class_order": [int(c) for c in base.classes_],
                        "train_range": ranges["train"], "calibration_range": ranges["calibration"], "calibration_method": "sigmoid",
                        "threshold": None, "selection_metrics": None, "package_versions": env["package_versions"],
                        "artifact_sha256": {"base_pipeline.joblib": file_hash(model_dir / "base_pipeline.joblib")},
                        "fit_seconds": time.perf_counter() - begin, "best_inner_cv_ap": float(search.cv_results_["mean_test_score"][search.best_index_]),
                        "inner_cv_std_ap": float(search.cv_results_["std_test_score"][search.best_index_]),
                        "matrix_bytes_float64": len(train) * len(transformed) * 8,
                        "fit_warnings": [str(w.message) for w in captured]}
            write_json(model_dir / "model_metadata.json", metadata)
            for i, parameters in enumerate(search.cv_results_["params"]):
                results.append({"model_id": model_id, "parameters": json.dumps(parameters, sort_keys=True),
                                "mean_ap": float(search.cv_results_["mean_test_score"][i]), "std_ap": float(search.cv_results_["std_test_score"][i]),
                                "selected": i == search.best_index_, **{f"fold_{j + 1}_ap": float(search.cv_results_[f"split{j}_test_score"][i]) for j in range(len(folds))}})
            write_csv(path / "cv_results.csv", pd.DataFrame(results))
            append_log(path, {"model_id": model_id, "state": "base_fitted", "duration_seconds": metadata["fit_seconds"]})
        write_json(path / "feature_manifest.json", {"feature_set": config["features"]["set"], "raw_feature_names": raw_features,
                                                   "excluded": ["duration", "campaign", "y", "row_id"], "pdays_transform": "previously_contacted=(pdays!=999); pdays_value=0 if 999 else pdays",
                                                   "transformed_feature_names": transformed})
        hashes = manifest["integrity"] | {name: file_hash(path / name) for name in ("cv_results.csv", "feature_manifest.json")}
        hashes.update({f"models/{model_id}/{filename}": file_hash(path / "models" / model_id / filename)
                       for model_id in config["models"] for filename in ("base_pipeline.joblib", "model_metadata.json")})
        update_manifest(path, status="trained", training_seconds=time.perf_counter() - started, integrity=hashes)
    except Exception as error:
        mark_failure(path, error)
        raise
    return run_id

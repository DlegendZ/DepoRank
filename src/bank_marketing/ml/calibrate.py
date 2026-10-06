import time

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from bank_marketing.ml.models import assert_binary
from bank_marketing.ml.registry import Registry
from bank_marketing.ml.train import load_run_data, mark_failure, require_stage, save_model, update_manifest
from bank_marketing.storage import append_log, file_hash, read_json, write_json


def calibrate_models(run_id, root="artifacts"):
    path = Registry(root).run_path(run_id)
    require_stage(path, "trained")
    data, config = load_run_data(path)
    ranges = read_json(path / "split_manifest.json")["ranges"]
    calibration = data.iloc[slice(*ranges["calibration"])]
    y = calibration.y.map({"no": 0, "yes": 1}).to_numpy()
    assert_binary(y)
    try:
        for model_id in config["models"]:
            print(f"Calibration sigmoid FrozenEstimator: {model_id}", flush=True)
            model_dir = path / "models" / model_id
            metadata = read_json(model_dir / "model_metadata.json")
            before = file_hash(model_dir / "base_pipeline.joblib")
            base = joblib.load(model_dir / "base_pipeline.joblib")
            started = time.perf_counter()
            calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
            calibrated.fit(calibration[metadata["raw_feature_names"]], y)
            save_model(model_dir / "calibrated_pipeline.joblib", calibrated)
            if file_hash(model_dir / "base_pipeline.joblib") != before:
                raise RuntimeError("Frozen base artifact changed")
            metadata["artifact_sha256"]["calibrated_pipeline.joblib"] = file_hash(model_dir / "calibrated_pipeline.joblib")
            metadata["calibration_seconds"] = time.perf_counter() - started
            write_json(model_dir / "model_metadata.json", metadata)
            append_log(path, {"model_id": model_id, "state": "calibrated", "duration_seconds": metadata["calibration_seconds"]})
        manifest = read_json(path / "run_manifest.json")
        integrity = manifest["integrity"] | {f"models/{model_id}/{filename}": file_hash(path / "models" / model_id / filename)
                                            for model_id in config["models"] for filename in ("calibrated_pipeline.joblib", "model_metadata.json")}
        update_manifest(path, status="calibrated", integrity=integrity)
    except Exception as error:
        mark_failure(path, error)
        raise
    return run_id

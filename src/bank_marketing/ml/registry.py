from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import math
from pathlib import Path
import platform
import re

import joblib

from bank_marketing.schemas import DomainError
from bank_marketing.storage import file_hash, object_hash, package_versions, read_json, write_json


def _artifact_errors(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError) as error:
            raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Artefak registry tidak lengkap atau tidak valid.") from error
    return checked
def _read_mapping(path):
    value = read_json(path)
    if not isinstance(value, dict):
        raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Artefak JSON harus mapping.")
    return value
def _artifact_path(base, name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Path artefak tidak valid.")
    relative = Path(name)
    target = (base / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(base.resolve()):
        raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Path artefak keluar dari run.")
    return target
def _verify_hashes(base, hashes):
    if not isinstance(hashes, dict):
        raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Daftar hash artefak harus mapping.")
    for name, expected in hashes.items():
        target = _artifact_path(base, name)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or not target.is_file() or file_hash(target) != expected:
            raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Integritas artefak gagal.")
@dataclass
class ModelRef:
    run_id: str
    model_id: str
    feature_set: str
    threshold: float
    model: object
    metadata: dict
    model_hash: str


class Registry:
    def __init__(self, root="artifacts"):
        self.root = Path(root).resolve()

    def run_path(self, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Run ID tidak valid.")
        return _artifact_path(self.root, f"runs/{run_id}")

    @_artifact_errors
    def verify_run(self, run_id, require_lock=True):
        path = self.run_path(run_id)
        if not (path / "run_manifest.json").is_file():
            raise DomainError("MODEL_NOT_READY", "Run belum tersedia.")
        manifest = _read_mapping(path / "run_manifest.json")
        if manifest.get("schema_version") != "1.0" or manifest.get("run_id") != run_id:
            raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Schema manifest tidak cocok.")
        _verify_hashes(path, manifest.get("integrity", {}))
        if require_lock:
            if not (path / "selection_lock.json").exists():
                raise DomainError("TEST_LOCK_REQUIRED", "Selection harus dikunci terlebih dahulu.")
            lock = _read_mapping(path / "selection_lock.json")
            if lock.get("schema_version") != "1.0" or lock.get("run_id") != run_id or lock.get("feature_set") != manifest.get("feature_set"):
                raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Schema/identitas selection lock tidak cocok.")
            digest = lock.get("lock_sha256")
            if digest != object_hash({k: v for k, v in lock.items() if k != "lock_sha256"}):
                raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Hash selection lock tidak cocok.")
            if manifest.get("status") not in ("locked", "tested"):
                raise DomainError("MODEL_NOT_READY", "Run belum locked/tested.")
            checks = {"dataset_sha256": "dataset_sha256", "config_sha256": "config_sha256", "split_manifest_sha256": "split_manifest_sha256"}
            for key, field in checks.items():
                if lock.get(key) != manifest.get(field):
                    raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Provenance lock tidak cocok.")
            if file_hash(path / "selection_metrics.csv") != lock["selection_metrics_hash"]:
                raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Selection metrics telah berubah.")
            candidates = lock["candidate_model_hashes"]
            if not isinstance(candidates, dict) or not candidates or lock["winner_model_id"] not in candidates:
                raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Winner/kandidat lock tidak valid.")
            if lock["feature_set"] not in ("core_v1", "context_v1") or set(lock["thresholds_by_model"]) != set(candidates):
                raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Feature set/threshold lock tidak valid.")
            for model_id, hashes in candidates.items():
                if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", model_id):
                    raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Model ID tidak valid.")
                if not isinstance(hashes, dict) or set(hashes) != {"base_pipeline.joblib", "calibrated_pipeline.joblib", "model_metadata.json"}:
                    raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Hash model wajib tidak lengkap.")
                threshold = lock["thresholds_by_model"][model_id]
                if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
                    raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Threshold terkunci tidak valid.")
                _verify_hashes(_artifact_path(path, f"models/{model_id}"), hashes)
            if manifest["status"] == "tested":
                evaluation = _read_mapping(path / "test_evaluation.json")
                if evaluation["selection_lock_sha256"] != digest:
                    raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Test memakai lock berbeda.")
                _verify_hashes(path, evaluation["result_hashes"])
        return manifest

    @_artifact_errors
    def load(self, run_id, model_id=None):
        path = self.run_path(run_id)
        manifest = self.verify_run(run_id)
        lock = _read_mapping(path / "selection_lock.json")
        model_id = model_id or lock["winner_model_id"]
        if model_id not in lock["candidate_model_hashes"]:
            raise DomainError("MODEL_NOT_READY", "Model bukan kandidat terkunci.")
        metadata = _read_mapping(path / "models" / model_id / "model_metadata.json")
        if metadata.get("schema_version") != "1.0" or metadata["feature_set"] != lock["feature_set"]:
            raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Metadata model tidak cocok.")
        versions = package_versions()
        for name in ("scikit-learn", "numpy", "pandas", "joblib"):
            if metadata["package_versions"].get(name) != versions[name]:
                raise DomainError("ENVIRONMENT_MISMATCH", "Gunakan requirements.lock environment model.")
        if manifest["python_version"] != platform.python_version():
            raise DomainError("ENVIRONMENT_MISMATCH", "Versi Python berbeda dari environment model.")
        expected = lock["candidate_model_hashes"][model_id]["calibrated_pipeline.joblib"]
        if metadata["artifact_sha256"]["calibrated_pipeline.joblib"] != expected:
            raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Metadata hash berbeda dari lock.")
        model = joblib.load(path / "models" / model_id / "calibrated_pipeline.joblib")
        if list(model.classes_) != metadata["probability_class_order"]:
            raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Urutan kelas berbeda.")
        return ModelRef(run_id, model_id, metadata["feature_set"], float(lock["thresholds_by_model"][model_id]), model, metadata, expected)

    @_artifact_errors
    def active(self):
        pointer = self.root / "active_run.json"
        if not pointer.exists():
            raise DomainError("MODEL_NOT_READY", "Belum ada run aktif. Jalankan CLI ingest sampai activate.")
        active = _read_mapping(pointer)
        lock = _read_mapping(self.run_path(active["run_id"]) / "selection_lock.json")
        if active["winner_model_id"] != lock["winner_model_id"] or active["selection_lock_sha256"] != lock["lock_sha256"]:
            raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Pointer aktif tidak cocok dengan winner terkunci.")
        return self.load(active["run_id"], active["winner_model_id"])

    def activate(self, run_id):
        ref = self.load(run_id)
        lock = _read_mapping(self.run_path(run_id) / "selection_lock.json")
        write_json(self.root / "active_run.json", {"run_id": run_id, "winner_model_id": ref.model_id, "selection_lock_sha256": lock["lock_sha256"]})
        return ref

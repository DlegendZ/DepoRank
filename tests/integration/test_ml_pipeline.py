import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from bank_marketing.config import load_config
from bank_marketing.features.preprocessing import build_pipeline
from bank_marketing.ml.evaluate import positive_probabilities
from bank_marketing.ml.final import evaluate_final
from bank_marketing.ml.models import MODEL_CLASSES, estimator_for
from bank_marketing.ml.registry import Registry
from bank_marketing.schemas import DomainError
from bank_marketing.storage import atomic_bytes, file_hash, object_hash, package_versions, write_json
import platform


@pytest.fixture
def profiles():
    rows = []
    for i in range(80):
        rows.append(dict(age=20 + i % 45, job="admin." if i % 2 else "technician", marital="single" if i % 3 else "married",
                         education="high.school", default="no", housing="yes", loan="no", pdays=999 if i % 2 else 3,
                         previous=i % 2, poutcome="nonexistent" if i % 2 else "failure"))
    return pd.DataFrame(rows), np.asarray([int(i % 3 == 0) for i in range(80)])


def test_all_five_real_estimators_and_frozen_calibration(profiles):
    X, y = profiles
    config = load_config("configs/experiment.yaml")
    for model_id in MODEL_CLASSES:
        estimator = estimator_for(model_id, config)
        base = build_pipeline(estimator, scale=model_id in ("logistic_regression", "gaussian_nb"))
        base.fit(X.iloc[:50], y[:50])
        before = positive_probabilities(base, X.iloc[:50]).copy()
        categories = [a.copy() for a in base.named_steps["preprocessor"].named_transformers_["categorical"].categories_]
        calibration = X.iloc[50:65].copy()
        calibration["job"] = "student"
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(calibration, y[50:65])
        np.testing.assert_allclose(positive_probabilities(base, X.iloc[:50]), before, atol=0, rtol=0)
        after_categories = base.named_steps["preprocessor"].named_transformers_["categorical"].categories_
        for a, b in zip(categories, after_categories):
            np.testing.assert_array_equal(a, b)
        assert "student" not in after_categories[0]
        buffer = io.BytesIO()
        joblib.dump(calibrated, buffer)
        buffer.seek(0)
        loaded = joblib.load(buffer)
        np.testing.assert_allclose(positive_probabilities(calibrated, X.iloc[65:]), positive_probabilities(loaded, X.iloc[65:]), atol=1e-8)
        assert set(base.named_steps["preprocessor"].get_feature_names_out()).isdisjoint({"duration", "campaign", "y", "row_id"})


def test_fold_encoder_and_scaler_only_fit_training(profiles):
    X, y = profiles
    config = load_config("configs/experiment.yaml")
    train = X.iloc[:30].copy()
    future = X.iloc[30:].copy()
    future["age"] = 120
    future["job"] = "student"
    base = build_pipeline(estimator_for("logistic_regression", config), scale=True).fit(train, y[:30])
    scaler = base.named_steps["preprocessor"].named_transformers_["numeric"]
    assert scaler.mean_[0] == pytest.approx(train.age.mean())
    assert scaler.mean_[0] != pytest.approx(pd.concat([train, future]).age.mean())
    assert "student" not in base.named_steps["preprocessor"].named_transformers_["categorical"].categories_[0]
    assert np.isfinite(positive_probabilities(base, future)).all()


def test_final_test_requires_lock_and_no_path_traversal(tmp_path):
    registry = Registry(tmp_path)
    path = registry.run_path("fixture")
    write_json(path / "run_manifest.json", {"schema_version": "1.0", "run_id": "fixture", "status": "selected", "integrity": {}})
    with pytest.raises(DomainError) as failure:
        evaluate_final("fixture", root=tmp_path)
    assert failure.value.code == "TEST_LOCK_REQUIRED"
    with pytest.raises(DomainError):
        registry.run_path("../escape")


def build_registry_fixture(tmp_path, profiles):
    X, y = profiles
    config = load_config("configs/experiment.yaml")
    base = build_pipeline(estimator_for("logistic_regression", config), scale=True).fit(X.iloc[:50], y[:50])
    model = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(X.iloc[50:65], y[50:65])
    registry = Registry(tmp_path)
    path = registry.run_path("fixture")
    model_dir = path / "models/logistic_regression"
    model_dir.mkdir(parents=True)
    for name, obj in (("base_pipeline.joblib", base), ("calibrated_pipeline.joblib", model)):
        content = io.BytesIO()
        joblib.dump(obj, content)
        atomic_bytes(model_dir / name, content.getvalue())
    metadata = {"schema_version": "1.0", "feature_set": "core_v1", "package_versions": package_versions(),
                "probability_class_order": [0, 1], "artifact_sha256": {"calibrated_pipeline.joblib": file_hash(model_dir / "calibrated_pipeline.joblib")}}
    write_json(model_dir / "model_metadata.json", metadata)
    atomic_bytes(path / "selection_metrics.csv", b"model_id,average_precision\nlogistic_regression,0.5\n")
    hashes = {name: file_hash(model_dir / name) for name in ("base_pipeline.joblib", "calibrated_pipeline.joblib", "model_metadata.json")}
    lock = {"schema_version": "1.0", "run_id": "fixture", "dataset_sha256": "d", "config_sha256": "c", "split_manifest_sha256": "s", "winner_model_id": "logistic_regression",
            "candidate_model_hashes": {"logistic_regression": hashes}, "selection_metrics_hash": file_hash(path / "selection_metrics.csv"),
            "feature_set": "core_v1", "thresholds_by_model": {"logistic_regression": 0.6}}
    lock["lock_sha256"] = object_hash(lock)
    write_json(path / "selection_lock.json", lock)
    write_json(path / "run_manifest.json", {"schema_version": "1.0", "run_id": "fixture", "status": "locked", "integrity": {},
                                           "dataset_sha256": "d", "config_sha256": "c", "split_manifest_sha256": "s", "feature_set": "core_v1", "python_version": platform.python_version()})
    return registry, path, model


def test_registry_trusted_loading_and_tamper_stop(tmp_path, profiles):
    registry, path, model = build_registry_fixture(tmp_path, profiles)
    ref = registry.activate("fixture")
    assert registry.active().threshold == 0.6
    np.testing.assert_allclose(positive_probabilities(ref.model, profiles[0]), positive_probabilities(model, profiles[0]), atol=1e-8)
    with (path / "models/logistic_regression/calibrated_pipeline.joblib").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(DomainError) as failure:
        registry.load("fixture")
    assert failure.value.code == "ARTIFACT_INTEGRITY_ERROR"


def test_registry_environment_mismatch_stop(tmp_path, profiles):
    registry, path, _ = build_registry_fixture(tmp_path, profiles)
    from bank_marketing.storage import read_json
    manifest = read_json(path / "run_manifest.json")
    manifest["python_version"] = "0.0.0"
    write_json(path / "run_manifest.json", manifest)
    with pytest.raises(DomainError) as failure:
        registry.load("fixture")
    assert failure.value.code == "ENVIRONMENT_MISMATCH"


def test_lock_self_hash_tamper_stop(tmp_path, profiles):
    registry, path, _ = build_registry_fixture(tmp_path, profiles)
    from bank_marketing.storage import read_json
    lock = read_json(path / "selection_lock.json")
    lock["thresholds_by_model"]["logistic_regression"] = 0.1
    write_json(path / "selection_lock.json", lock)
    with pytest.raises(DomainError) as failure:
        registry.verify_run("fixture")
    assert failure.value.code == "ARTIFACT_INTEGRITY_ERROR"


def test_test_overwrite_guard_without_reading_holdout(tmp_path, profiles):
    registry, path, _ = build_registry_fixture(tmp_path, profiles)
    write_json(path / "test_attempt.json", {"started": True})
    with pytest.raises(DomainError) as failure:
        evaluate_final("fixture", root=tmp_path)
    assert failure.value.code == "TEST_ALREADY_EVALUATED"


def test_prelock_stage_hash_detects_tampering(tmp_path):
    registry = Registry(tmp_path)
    path = registry.run_path("fixture")
    atomic_bytes(path / "models/base.joblib", b"trusted-generated-placeholder-not-loaded")
    digest = file_hash(path / "models/base.joblib")
    write_json(path / "run_manifest.json", {"schema_version": "1.0", "run_id": "fixture", "status": "trained", "integrity": {"models/base.joblib": digest}})
    registry.verify_run("fixture", require_lock=False)
    atomic_bytes(path / "models/base.joblib", b"tampered")
    with pytest.raises(DomainError) as failure:
        registry.verify_run("fixture", require_lock=False)
    assert failure.value.code == "ARTIFACT_INTEGRITY_ERROR"


def test_atomic_exclusive_preserves_existing_file(tmp_path):
    path = tmp_path / "lock.json"
    write_json(path, {"winner": "original"}, exclusive=True)
    with pytest.raises(FileExistsError):
        write_json(path, {"winner": "replacement"}, exclusive=True)
    from bank_marketing.storage import read_json
    assert read_json(path) == {"winner": "original"}
    assert not list(tmp_path.glob(".pending_*"))



def test_two_fits_reproduce_scores_with_same_seed(profiles):
    X, y = profiles
    config = load_config("configs/experiment.yaml")
    for model_id in MODEL_CLASSES:
        outputs = []
        for _ in range(2):
            base = build_pipeline(estimator_for(model_id, config), scale=model_id in ("logistic_regression", "gaussian_nb"))
            base.fit(X.iloc[:50], y[:50])
            calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(X.iloc[50:65], y[50:65])
            outputs.append(positive_probabilities(calibrated, X.iloc[65:]))
        np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-8, rtol=0)


def test_training_rejects_modified_order_even_with_valid_raw_hash(tmp_path):
    source = Path("data/raw/bank-additional-full.csv")
    if not source.exists():
        pytest.skip("Official raw not ingested in this environment")
    from bank_marketing.ml.train import train_models
    data = pd.read_csv(source, sep=";").iloc[::-1].reset_index(drop=True)
    config = load_config("configs/experiment.yaml")
    with pytest.raises(DomainError) as failure:
        train_models(data, config, root=tmp_path)
    assert failure.value.code == "DATASET_HASH_MISMATCH"
    assert not list(tmp_path.glob("runs/*"))


@pytest.mark.parametrize('content', [b'{', b'[]', b'{}', b'{"run_id":null}'])
def test_malformed_active_pointer_is_domain_error(tmp_path, content):
    registry = Registry(tmp_path)
    atomic_bytes(tmp_path / 'active_run.json', content)
    with pytest.raises(DomainError) as failure:
        registry.active()
    assert failure.value.code == 'ARTIFACT_INTEGRITY_ERROR'
@pytest.mark.parametrize('content', [b'{', b'[]', b'{"schema_version":"1.0","run_id":"fixture","integrity":null}'])
def test_malformed_run_manifest_is_domain_error(tmp_path, content):
    registry = Registry(tmp_path)
    atomic_bytes(registry.run_path('fixture') / 'run_manifest.json', content)
    with pytest.raises(DomainError) as failure:
        registry.verify_run('fixture', require_lock=False)
    assert failure.value.code == 'ARTIFACT_INTEGRITY_ERROR'
def test_registry_rejects_artifact_path_escape(tmp_path):
    outside = tmp_path / 'outside.txt'
    atomic_bytes(outside, b'outside run')
    registry = Registry(tmp_path)
    write_json(registry.run_path('fixture') / 'run_manifest.json', {
        'schema_version': '1.0', 'run_id': 'fixture', 'status': 'trained',
        'integrity': {'../../outside.txt': file_hash(outside)},
    })
    with pytest.raises(DomainError) as failure:
        registry.verify_run('fixture', require_lock=False)
    assert failure.value.code == 'ARTIFACT_INTEGRITY_ERROR'
@pytest.mark.parametrize('threshold', [None, True, -0.1, 1.1, '0.5'])
def test_invalid_locked_threshold_stops_loading(tmp_path, profiles, threshold):
    from bank_marketing.storage import read_json
    registry, path, _ = build_registry_fixture(tmp_path, profiles)
    lock = read_json(path / 'selection_lock.json')
    lock['thresholds_by_model']['logistic_regression'] = threshold
    lock['lock_sha256'] = object_hash({k: v for k, v in lock.items() if k != 'lock_sha256'})
    write_json(path / 'selection_lock.json', lock)
    with pytest.raises(DomainError) as failure:
        registry.load('fixture')
    assert failure.value.code == 'ARTIFACT_INTEGRITY_ERROR'
def test_lock_must_hash_all_required_model_files(tmp_path, profiles):
    from bank_marketing.storage import read_json
    registry, path, _ = build_registry_fixture(tmp_path, profiles)
    lock = read_json(path / 'selection_lock.json')
    lock['candidate_model_hashes']['logistic_regression'].pop('model_metadata.json')
    lock['lock_sha256'] = object_hash({k: v for k, v in lock.items() if k != 'lock_sha256'})
    write_json(path / 'selection_lock.json', lock)
    with pytest.raises(DomainError) as failure:
        registry.load('fixture')
    assert failure.value.code == 'ARTIFACT_INTEGRITY_ERROR'
def test_interrupted_final_attempt_cannot_be_retried(tmp_path, profiles):
    from bank_marketing.storage import read_json
    registry, path, _ = build_registry_fixture(tmp_path, profiles)
    write_json(path / 'test_attempt.json', {'started': True})
    manifest = read_json(path / 'run_manifest.json')
    manifest['status'] = 'failed'
    write_json(path / 'run_manifest.json', manifest)
    with pytest.raises(DomainError) as failure:
        evaluate_final('fixture', root=tmp_path)
    assert failure.value.code == 'TEST_ALREADY_EVALUATED'
def test_direct_training_api_cannot_bypass_baseline_config(tmp_path, profiles):
    from bank_marketing.ml.train import train_models
    config = load_config('configs/experiment.yaml')
    config['features']['forbidden'] = []
    with pytest.raises(DomainError) as failure:
        train_models(profiles[0], config, root=tmp_path)
    assert failure.value.code == 'SCHEMA_INVALID'
    assert not list(tmp_path.glob('runs/*'))
@pytest.mark.parametrize('field,value', [('schema_version', '9.9'), ('run_id', 'other'), ('feature_set', 'context_v1')])
def test_lock_identity_must_match_manifest(tmp_path, profiles, field, value):
    from bank_marketing.storage import read_json
    registry, path, _ = build_registry_fixture(tmp_path, profiles)
    lock = read_json(path / 'selection_lock.json')
    lock.update(schema_version='1.0', run_id='fixture')
    lock[field] = value
    lock['lock_sha256'] = object_hash({k: v for k, v in lock.items() if k != 'lock_sha256'})
    write_json(path / 'selection_lock.json', lock)
    with pytest.raises(DomainError) as failure:
        registry.verify_run('fixture')
    assert failure.value.code == 'ARTIFACT_INTEGRITY_ERROR'
def test_prelock_stages_preserve_metadata_integrity(tmp_path, profiles):
    from bank_marketing.ml.calibrate import calibrate_models
    from bank_marketing.ml.selection import select_model, freeze_selection
    from bank_marketing.ml.train import save_model
    from bank_marketing.storage import read_json
    X, y = profiles
    registry = Registry(tmp_path)
    path = registry.run_path('stages')
    data = X.assign(y=np.where(y, 'yes', 'no'))
    source = tmp_path / 'fixture.csv'
    source.write_text(data.to_csv(sep=';', index=False), encoding='utf-8')
    config = load_config('configs/experiment.yaml')
    config['dataset']['path'] = str(source)
    config['models'] = ['logistic_regression']
    import yaml
    atomic_bytes(path / 'config_resolved.yaml', yaml.safe_dump(config).encode())
    write_json(path / 'split_manifest.json', {'ranges': {'train': [0, 50], 'calibration': [50, 65], 'selection': [65, 80], 'test': [80, 80]}})
    base = build_pipeline(estimator_for('logistic_regression', config), scale=True).fit(X.iloc[:50], y[:50])
    directory = path / 'models/logistic_regression'
    save_model(directory / 'base_pipeline.joblib', base)
    write_json(directory / 'model_metadata.json', {
        'schema_version': '1.0', 'feature_set': 'core_v1', 'raw_feature_names': list(X.columns),
        'artifact_sha256': {'base_pipeline.joblib': file_hash(directory / 'base_pipeline.joblib')},
        'fit_seconds': 0.01, 'best_inner_cv_ap': 0.5,
    })
    names = ['config_resolved.yaml', 'split_manifest.json', 'models/logistic_regression/base_pipeline.joblib', 'models/logistic_regression/model_metadata.json']
    write_json(path / 'run_manifest.json', {
        'schema_version': '1.0', 'run_id': 'stages', 'feature_set': 'core_v1', 'status': 'trained',
        'dataset_sha256': file_hash(source), 'config_sha256': object_hash(config),
        'split_manifest_sha256': file_hash(path / 'split_manifest.json'),
        'integrity': {name: file_hash(path / name) for name in names},
    })
    calibrate_models('stages', root=tmp_path)
    registry.verify_run('stages', require_lock=False)
    select_model('stages', root=tmp_path)
    manifest = registry.verify_run('stages', require_lock=False)
    assert {'selection_result.json', 'selection_metrics.csv'} <= set(manifest['integrity'])
    lock = freeze_selection('stages', root=tmp_path)
    assert lock['winner_model_id'] == 'logistic_regression'
    registry.verify_run('stages')
    metadata = read_json(directory / 'model_metadata.json')
    metadata['raw_feature_names'] = ['duration']
    write_json(directory / 'model_metadata.json', metadata)
    with pytest.raises(DomainError) as failure:
        registry.verify_run('stages')
    assert failure.value.code == 'ARTIFACT_INTEGRITY_ERROR'

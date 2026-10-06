from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from bank_marketing.ml.evaluate import evaluate_scores, positive_probabilities, threshold_for_f1
from bank_marketing.ml.registry import Registry
from bank_marketing.ml.train import load_run_data, mark_failure, require_stage, update_manifest, write_csv
from bank_marketing.storage import file_hash, object_hash, read_json, utc_now, write_json


def choose_winner(candidates, delta=0.005):
    maximum = max(float(c["average_precision"]) for c in candidates)
    tied = [c for c in candidates if maximum - float(c["average_precision"]) <= delta + 1e-12]
    winner = min(tied, key=lambda c: (-c["precision_at_k"], c["brier_score"], c["inference_latency"], c["model_id"]))
    return {"winner_model_id": winner["model_id"], "maximum_ap": maximum,
            "practical_tie_delta": delta, "eligible_candidates": [c["model_id"] for c in tied],
            "reason": "AP practical tie <=0.005; precision@10% desc, Brier asc, median inference latency asc, model_id asc",
            "candidates": candidates}


def benchmark_inference(model, X, repetitions=20):
    positive_probabilities(model, X)
    samples = []
    for _ in range(repetitions):
        begin = time.perf_counter()
        positive_probabilities(model, X)
        samples.append(time.perf_counter() - begin)
    return float(np.median(samples))


def save_evaluation(path, split, model_id, y, p, threshold, row_ids, raw=None):
    report = evaluate_scores(y, p, threshold, row_ids=row_ids)
    report.update({"split": split, "model_id": model_id, "run_id": path.name, "feature_set": read_json(path / "run_manifest.json")["feature_set"]})
    write_json(path / "evaluations" / f"{split}_{model_id}.json", report)
    predictions = pd.DataFrame({"row_id": row_ids, "y": y, "p_yes": p, "predicted_label": np.where(p >= threshold, "yes", "no")})
    if raw is not None:
        predictions["raw_p_yes_before_calibration"] = raw
    write_csv(path / "predictions" / f"{split}_{model_id}.csv", predictions)
    return report


def baseline_reports(path, split, data, ranges):
    train = data.iloc[slice(*ranges["train"])]
    evaluation = data.iloc[slice(*ranges[split])]
    y = evaluation.y.map({"no": 0, "yes": 1}).to_numpy()
    train_y = train.y.map({"no": 0, "yes": 1}).to_numpy()
    dummy = DummyClassifier(strategy="prior").fit(np.zeros((len(train), 1)), train_y)
    p = positive_probabilities(dummy, np.zeros((len(evaluation), 1)))
    row_ids = np.arange(*ranges[split])
    prior = save_evaluation(path, split, "dummy_prior", y, p, 0.5, row_ids)
    random_scores = np.random.default_rng(42).random(len(y))
    ranking = save_evaluation(path, split, "random_ranking", y, random_scores, 0.5, row_ids)
    ranking["note"] = "Ranking acak seed=42; scores bukan probabilitas. Metrik probabilitas/classification N/A."
    for key in ("brier_score", "precision", "recall", "f1", "accuracy", "threshold"):
        ranking["metrics"][key] = None
        ranking["metrics"]["reason_codes"][key] = "RANDOM_RANKING_NOT_PROBABILITY"
    ranking["confusion_matrix"] = None
    ranking["curves"]["calibration"] = []
    write_json(path / "evaluations" / f"{split}_random_ranking.json", ranking)
    return [{"model_id": "dummy_prior", **prior["metrics"]}, {"model_id": "random_ranking", **ranking["metrics"]}]


def select_model(run_id, root="artifacts"):
    path = Registry(root).run_path(run_id)
    require_stage(path, "calibrated")
    data, config = load_run_data(path)
    ranges = read_json(path / "split_manifest.json")["ranges"]
    selection = data.iloc[slice(*ranges["selection"])]
    y = selection.y.map({"no": 0, "yes": 1}).to_numpy()
    row_ids = np.arange(*ranges["selection"])
    rows = []
    try:
        for model_id in config["models"]:
            model_dir = path / "models" / model_id
            model = joblib.load(model_dir / "calibrated_pipeline.joblib")
            base = joblib.load(model_dir / "base_pipeline.joblib")
            metadata = read_json(model_dir / "model_metadata.json")
            X = selection[metadata["raw_feature_names"]]
            p, raw = positive_probabilities(model, X), positive_probabilities(base, X)
            threshold = threshold_for_f1(y, p)
            report = save_evaluation(path, "selection", model_id, y, p, threshold, row_ids, raw)
            report["metrics_at_0_5"] = evaluate_scores(y, p, 0.5, row_ids)["metrics"]
            write_json(path / "evaluations" / f"selection_{model_id}.json", report)
            latency = benchmark_inference(model, X)
            row = {"model_id": model_id, **report["metrics"], "inference_latency": latency,
                   "fit_seconds": metadata["fit_seconds"], "inner_cv_ap": metadata["best_inner_cv_ap"]}
            rows.append(row)
            metadata.update(threshold=threshold, selection_metrics=report["metrics"], inference_latency=latency)
            write_json(model_dir / "model_metadata.json", metadata)
        policy = choose_winner(rows, config["selection"]["practical_tie_delta"])
        write_json(path / "selection_result.json", policy)
        write_csv(path / "selection_metrics.csv", pd.DataFrame(rows + baseline_reports(path, "selection", data, ranges)))
        integrity = read_json(path / "run_manifest.json")["integrity"] | {
            name: file_hash(path / name) for name in ("selection_result.json", "selection_metrics.csv")}
        integrity.update({f"models/{model_id}/model_metadata.json": file_hash(path / "models" / model_id / "model_metadata.json")
                          for model_id in config["models"]})
        update_manifest(path, status="selected", winner_model_id=policy["winner_model_id"], integrity=integrity)
    except Exception as error:
        mark_failure(path, error)
        raise
    return policy


def freeze_selection(run_id, root="artifacts"):
    path = Registry(root).run_path(run_id)
    manifest = require_stage(path, "selected")
    policy = read_json(path / "selection_result.json")
    _, config = load_run_data(path)
    hashes, thresholds = {}, {}
    for model_id in config["models"]:
        directory = path / "models" / model_id
        metadata = read_json(directory / "model_metadata.json")
        thresholds[model_id] = metadata["threshold"]
        hashes[model_id] = {name: file_hash(directory / name) for name in ("base_pipeline.joblib", "calibrated_pipeline.joblib", "model_metadata.json")}
    lock = {"schema_version": "1.0", "run_id": run_id, "locked_at_utc": utc_now(),
            "dataset_sha256": manifest["dataset_sha256"], "config_sha256": manifest["config_sha256"],
            "split_manifest_sha256": manifest["split_manifest_sha256"], "candidate_model_hashes": hashes,
            "winner_model_id": policy["winner_model_id"], "selection_policy_version": "1.0",
            "selection_metrics_hash": file_hash(path / "selection_metrics.csv"), "thresholds_by_model": thresholds,
            "q_default": config["capacity"]["default_fraction"], "feature_set": config["features"]["set"]}
    lock["lock_sha256"] = object_hash(lock)
    write_json(path / "selection_lock.json", lock, exclusive=True)
    integrity = manifest["integrity"] | {"selection_result.json": file_hash(path / "selection_result.json"), "selection_lock.json": file_hash(path / "selection_lock.json")}
    integrity.update({str(p.relative_to(path)).replace("\\", "/"): file_hash(p) for folder in ("evaluations", "predictions") for p in (path / folder).glob("selection_*")})
    update_manifest(path, status="locked", integrity=integrity)
    return lock

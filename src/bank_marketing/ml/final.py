from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from bank_marketing.ml.evaluate import evaluate_scores, positive_probabilities
from bank_marketing.ml.registry import Registry
from bank_marketing.ml.selection import baseline_reports, save_evaluation
from bank_marketing.ml.train import load_run_data, mark_failure, update_manifest, write_csv
from bank_marketing.schemas import DomainError
from bank_marketing.storage import file_hash, read_json, utc_now, write_json


def evaluate_final(run_id, root="artifacts"):
    registry = Registry(root)
    path = registry.run_path(run_id)
    if (path / "test_evaluation.json").exists() or (path / "test_attempt.json").exists():
        raise DomainError("TEST_ALREADY_EVALUATED", "Final test must not be rerun or overwritten.")
    registry.verify_run(run_id)
    lock = read_json(path / "selection_lock.json")
    data, config = load_run_data(path)
    ranges = read_json(path / "split_manifest.json")["ranges"]
    write_json(path / "test_attempt.json", {"run_id": run_id, "selection_lock_sha256": lock["lock_sha256"], "started_at_utc": utc_now()}, exclusive=True)
    test = data.iloc[slice(*ranges["test"])]
    y = test.y.map({"no": 0, "yes": 1}).to_numpy()
    row_ids = np.arange(*ranges["test"])
    rows = []
    started = time.perf_counter()
    try:
        for model_id in config["models"]:
            ref = registry.load(run_id, model_id)
            p = positive_probabilities(ref.model, test[ref.metadata["raw_feature_names"]])
            base = joblib.load(path / "models" / model_id / "base_pipeline.joblib")
            raw = positive_probabilities(base, test[ref.metadata["raw_feature_names"]])
            report = save_evaluation(path, "test", model_id, y, p, ref.threshold, row_ids, raw)
            report["metrics_at_0_5"] = evaluate_scores(y, p, 0.5, row_ids)["metrics"]
            write_json(path / "evaluations" / f"test_{model_id}.json", report)
            rows.append({"model_id": model_id, **report["metrics"], "is_selection_winner": model_id == lock["winner_model_id"]})
        write_csv(path / "test_metrics.csv", pd.DataFrame(rows + baseline_reports(path, "test", data, ranges)))
        result_files = [path / "test_metrics.csv"] + list((path / "predictions").glob("test_*.csv")) + list((path / "evaluations").glob("test_*.json"))
        report = {"run_id": run_id, "selection_lock_sha256": lock["lock_sha256"], "evaluated_at_utc": utc_now(),
                  "test_range": ranges["test"], "status": "complete", "evaluator_version": "1.0",
                  "result_hashes": {str(p.relative_to(path)).replace("\\", "/"): file_hash(p) for p in result_files},
                  "duration_seconds": time.perf_counter() - started, "winner_model_id": lock["winner_model_id"]}
        write_json(path / "test_evaluation.json", report, exclusive=True)
        update_manifest(path, status="tested", finished_at_utc=utc_now())
    except Exception as error:
        mark_failure(path, error)
        raise
    return report

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import psutil

from bank_marketing.ml.registry import Registry
from bank_marketing.ml.train import load_run_data
from bank_marketing.services.ranking import rank_records
from bank_marketing.services.scoring import score_records
from bank_marketing.storage import environment, read_json, utc_now, write_json


def benchmark_run(run_id, root="artifacts"):
    registry = Registry(root)
    path = registry.run_path(run_id)
    data, _ = load_run_data(path)
    start = time.perf_counter()
    ref = registry.load(run_id)
    startup = time.perf_counter() - start
    X = data.loc[:, ref.metadata["raw_feature_names"]]
    sample = X.iloc[[0]].copy()
    score_records(sample, ref)
    latencies = []
    for _ in range(20):
        begin = time.perf_counter()
        score_records(sample, ref)
        latencies.append(time.perf_counter() - begin)
    batch = X.iloc[:10000].copy()
    csv = batch.to_csv(index=False, sep=";").encode("utf-8")
    from bank_marketing.data.ingest import read_csv
    process = psutil.Process()
    memory = [process.memory_info().rss]
    stop = threading.Event()
    def monitor():
        while not stop.wait(0.005):
            memory.append(process.memory_info().rss)
    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    begin = time.perf_counter()
    try:
        parsed = read_csv(csv)
        result = rank_records(score_records(parsed, ref), {"q": 0.1})
        elapsed = time.perf_counter() - begin
    finally:
        stop.set()
        worker.join()
    manifest = read_json(path / "run_manifest.json")
    report = {"run_id": run_id, "model_id": ref.model_id, "measured_at_utc": utc_now(), "environment": environment(),
              "startup_registry_warmed_seconds": startup, "single_median_seconds_20_repetitions": float(np.median(latencies)),
              "batch_rows": len(result["rows"]), "batch_parse_validate_predict_rank_seconds": elapsed,
              "peak_process_rss_bytes_sampled": max(memory), "training_grid_seconds": manifest.get("training_seconds"),
              "targets": {"NFR_03_startup_10s": startup <= 10, "NFR_04_single_1s": float(np.median(latencies)) <= 1,
                          "NFR_05_batch_15s": elapsed <= 15, "NFR_05_memory_4GB": max(memory) <= 4 * 1024**3,
                          "NFR_06_training_45min": manifest.get("training_seconds", float("inf")) <= 2700},
              "notes": "Startup measures a warm-disk registry load, not cold Python/Streamlit startup. RSS sampling every 5 ms is not full allocation instrumentation."}
    write_json(path / "benchmark.json", report)
    return report

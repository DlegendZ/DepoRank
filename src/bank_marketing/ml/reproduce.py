import numpy as np
import pandas as pd

from bank_marketing.ml.evaluate import positive_probabilities
from bank_marketing.ml.registry import Registry
from bank_marketing.ml.train import load_run_data
from bank_marketing.schemas import DomainError
from bank_marketing.storage import read_json


def verify_prediction_parity(run_id, root="artifacts"):
    registry = Registry(root)
    path = registry.run_path(run_id)
    manifest = registry.verify_run(run_id)
    data, config = load_run_data(path)
    ranges = read_json(path / "split_manifest.json")["ranges"]
    checks = []
    for split in (["selection", "test"] if manifest["status"] == "tested" else ["selection"]):
        block = data.iloc[slice(*ranges[split])]
        for model_id in config["models"]:
            ref = registry.load(run_id, model_id)
            scores = positive_probabilities(ref.model, block[ref.metadata["raw_feature_names"]])
            saved = pd.read_csv(path / "predictions" / f"{split}_{model_id}.csv")
            difference = float(np.max(np.abs(scores - saved.p_yes.to_numpy())))
            if not np.array_equal(saved.row_id.to_numpy(), np.arange(*ranges[split])) or not np.allclose(scores, saved.p_yes, atol=1e-8, rtol=0):
                raise DomainError("ARTIFACT_INTEGRITY_ERROR", "Prediksi reproduksi berbeda dari baseline tersimpan.")
            checks.append({"split": split, "model_id": model_id, "max_absolute_difference": difference, "atol": 1e-8})
    return {"run_id": run_id, "status": "identical_within_tolerance", "checks": checks, "read_only": True}

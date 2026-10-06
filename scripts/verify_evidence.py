"""Compare saved predictions with the frozen models without repeating final test."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bank_marketing.ml.reproduce import verify_prediction_parity


if __name__ == "__main__":
    config = json.loads((ROOT / "configs/submission.json").read_text(encoding="utf-8"))
    result = verify_prediction_parity(config["run_id"], ROOT / "artifacts")
    (ROOT / "reports/reproduction_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

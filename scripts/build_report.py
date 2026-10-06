"""Refresh the English report and presentation data from the existing frozen run."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bank_marketing.reporting import generate_reports


if __name__ == "__main__":
    configuration = json.loads((ROOT / "configs/submission.json").read_text(encoding="utf-8"))
    print(json.dumps(generate_reports(configuration["run_id"], ROOT / "artifacts"), indent=2))

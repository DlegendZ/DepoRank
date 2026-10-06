"""Run tests and save observed counts for the report without claiming model quality."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="deporank_tests_") as temporary:
        xml = Path(temporary) / "tests.xml"
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={xml}"], cwd=ROOT)
        if not xml.exists():
            raise SystemExit(result.returncode or 1)
        suites = ET.parse(xml).getroot().iter("testsuite")
        totals = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
        seconds = 0.0
        for suite in suites:
            for key in totals:
                totals[key] += int(suite.attrib.get(key, 0))
            seconds += float(suite.attrib.get("time", 0))
        summary = {**totals, "passed": totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"], "seconds": seconds, "executed_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0], "command": "python -m pytest -q --junitxml=<temporary XML>", "note": "Software verification only; not model accuracy or production readiness."}
        output = ROOT / "reports/software_quality.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        raise SystemExit(result.returncode)

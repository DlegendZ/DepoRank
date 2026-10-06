"""Check anonymous public access; run again after publishing the GitHub commits."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def check(url):
    request = urllib.request.Request(url, headers={"User-Agent": "DepoRank-submission-check"}, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            accessible = 200 <= response.status < 300 and "github.com/login" not in response.url
            return {"url": url, "status": response.status, "resolved_url": response.url, "accessible_without_login": accessible}
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
        return {"url": url, "status": getattr(error, "code", None), "accessible_without_login": False, "error": str(error)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", action="store_true", help="Also check the report's official research/documentation links.")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/submission.json").read_text(encoding="utf-8"))
    repository = config["repository_url"].rstrip("/")
    urls = {
        "slides": repository + f"/raw/refs/heads/{config['branch']}/presentation/slides.pptx",
        "dataset": repository + f"/raw/refs/heads/{config['branch']}/data/raw/bank-additional-full.csv",
        "source_code": repository,
    }
    links = {name: check(url) for name, url in urls.items()}
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "authentication": "none", "all_accessible": all(item["accessible_without_login"] for item in links.values()), "links": links, "note": "HTTP access only. Also open/download in a signed-out browser. Unpushed files are not public."}
    if args.references:
        references = [
            "https://archive.ics.uci.edu/dataset/222/bank+marketing",
            "https://doi.org/10.1016/j.dss.2014.03.001",
            "https://scikit-learn.org/stable/modules/calibration.html",
            "https://scikit-learn.org/stable/modules/model_evaluation.html",
            "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html",
        ]
        result["references"] = [check(url) for url in references]
    output = ROOT / "reports/publication_checks.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["all_accessible"] else 1)

"""Build exactly four upload files from the committed project snapshot."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_archive(path, files, manifest):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, name in files:
            archive.write(source, name)
        archive.writestr("SUBMISSION_MANIFEST.json", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true", help="Include tracked local edits; commit all required new files before packaging.")
    args = parser.parse_args()
    changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True).strip()
    if changed and not args.allow_dirty:
        raise ValueError("Tracked changes remain. Commit them before packaging, or explicitly use --allow-dirty.")
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    files = []
    for name in filter(None, names):
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file():
            raise ValueError(f"Invalid source snapshot entry: {name}")
        files.append((source, name))
    for required in ("scripts/restore_demo.py", "demo/deporank_demo_run.zip", "README.md", "data/raw/bank-additional-full.csv", "presentation/slides.pptx", "reports/final_report.pdf"):
        if required not in names:
            raise ValueError(f"Required file is not committed/staged: {required}")
    quality = json.loads((ROOT / "reports/software_quality.json").read_text(encoding="utf-8"))
    if quality["failures"] or quality["errors"]:
        raise ValueError("Recorded software tests failed; fix and rerun check_quality.py.")
    config = json.loads((ROOT / "configs/submission.json").read_text(encoding="utf-8"))
    dataset = ROOT / "data/raw/bank-additional-full.csv"
    dataset_manifest = json.loads((ROOT / "data/manifests/dataset_manifest.json").read_text(encoding="utf-8"))
    if digest(dataset) != dataset_manifest["dataset_sha256"]:
        raise ValueError("Raw dataset hash changed; packaging stopped.")
    output = ROOT / "submission"
    output.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "presentation/slides.pptx", output / "DepoRank_Presentation.pptx")
    shutil.copyfile(ROOT / "reports/final_report.pdf", output / "DepoRank_Report.pdf")
    data_names = ["data/README.md", "data/raw/bank-additional-full.csv", "data/raw/bank-additional-names.txt", "data/manifests/dataset_manifest.json", "data/manifests/audit.json", "data/manifests/split_manifest.json"]
    data_files = [(ROOT / name, name.removeprefix("data/")) for name in data_names]
    write_archive(output / "DepoRank_Dataset.zip", data_files, {"variant": "bank-additional-full", "rows": 41188, "license": "CC BY 4.0", "files": {name: digest(source) for source, name in data_files}})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_manifest = {"commit": commit, "contains_tracked_local_edits": bool(changed), "run_id": config["run_id"], "files": {name: digest(source) for source, name in files}, "excludes": [".git", "virtual environments", "caches", "uploads", "private course references", "unpacked model artifacts", "submission archives"]}
    write_archive(output / "DepoRank_Source_Code.zip", files, source_manifest)
    expected = {"DepoRank_Presentation.pptx", "DepoRank_Report.pdf", "DepoRank_Dataset.zip", "DepoRank_Source_Code.zip"}
    extras = {path.name for path in output.iterdir()} - expected
    if extras:
        raise ValueError(f"Unexpected submission-folder files; inspect manually: {sorted(extras)}")
    print(json.dumps({"commit": commit, "files": {name: {"bytes": (output / name).stat().st_size, "sha256": digest(output / name)} for name in sorted(expected)}}, indent=2))

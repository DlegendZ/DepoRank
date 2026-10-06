"""Compress the verified academic run without changing hash-locked evidence."""

import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bank_marketing.ml.registry import Registry


if __name__ == "__main__":
    configuration = json.loads((ROOT / "configs/submission.json").read_text(encoding="utf-8"))
    registry = Registry(ROOT / "artifacts")
    manifest = registry.verify_run(configuration["run_id"])
    active = json.loads((ROOT / "artifacts/active_run.json").read_text(encoding="utf-8"))
    if active["run_id"] != configuration["run_id"]:
        raise ValueError("Active run and submission run differ.")
    run = registry.run_path(configuration["run_id"])
    files = [ROOT / "artifacts/active_run.json"] + sorted(source for source in run.rglob("*") if source.is_file() and "logs" not in source.relative_to(run).parts)
    output = ROOT / "demo/deporank_demo_run.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in files:
            entry = zipfile.ZipInfo(source.relative_to(ROOT).as_posix(), date_time=(2026, 10, 5, 14, 12, 42))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, source.read_bytes(), compresslevel=9)
    result = {"run_id": configuration["run_id"], "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "compressed_bytes": output.stat().st_size, "uncompressed_bytes": sum(source.stat().st_size for source in files), "files": len(files), "status": manifest["status"], "note": "Original hash-locked evidence preserved byte for byte. Charts carry English labels; run logs are omitted."}
    (ROOT / "demo/manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

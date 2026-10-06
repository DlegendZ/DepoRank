"""Restore the repository's checksum-pinned academic demo without retraining."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def restore(root=ROOT, bundle=None):
    root = Path(root).resolve()
    bundle = Path(bundle or ROOT / "demo/deporank_demo_run.zip").resolve()
    expected = json.loads((ROOT / "demo/manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != expected["sha256"]:
        raise ValueError("Demo archive checksum mismatch; restore stopped.")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="deporank_restore_", dir=root) as temporary:
        temporary = Path(temporary)
        with zipfile.ZipFile(bundle) as archive:
            total = 0
            names = set()
            for member in archive.infolist():
                target = (temporary / member.filename).resolve()
                if (not target.is_relative_to(temporary) or "\\" in member.orig_filename
                        or member.is_dir() or ":" in member.filename
                        or not member.filename.startswith("artifacts/") or member.filename in names
                        or (member.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ValueError("Unsafe demo archive member; restore stopped.")
                names.add(member.filename)
                total += member.file_size
                if total > 250_000_000:
                    raise ValueError("Demo archive exceeds extraction size limit.")
            archive.extractall(temporary)
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        for source in files:
            target = (root / source.relative_to(temporary)).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Existing output path escapes the restore root.")
            if target.exists() and (not target.is_file() or target.read_bytes() != source.read_bytes()):
                raise FileExistsError(f"Existing artifact differs; no files changed: {target}")
        restored = 0
        for source in files:
            target = root / source.relative_to(temporary)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                restored += 1
    return {"run_id": expected["run_id"], "files_restored": restored, "root": str(root)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(restore(parser.parse_args().root), indent=2))

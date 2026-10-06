from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
from datetime import datetime, timezone

import psutil


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def object_hash(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path, content: bytes, exclusive=False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            # Hard-link publication is atomic and cannot replace an existing lock.
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value, exclusive=False) -> None:
    atomic_bytes(path, canonical_bytes(value), exclusive=exclusive)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def package_versions() -> dict:
    names = ("numpy", "pandas", "scikit-learn", "joblib", "streamlit", "plotly", "PyYAML", "pydantic")
    return {name: importlib.metadata.version(name) for name in names}


def environment() -> dict:
    return {"python_version": platform.python_version(), "package_versions": package_versions(),
            "os": platform.platform(), "cpu": platform.processor(), "cpu_count": psutil.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total}


def append_log(run_path, event: dict) -> None:
    path = Path(run_path) / "logs/run.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical_bytes({"at_utc": utc_now(), **event}) + b"\n")

"""Bounded CSV parsing and verified immutable official dataset ingest."""

import csv
import hashlib
import io
import json
import re
import stat
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
import zipfile

import pandas as pd

from bank_marketing.data.audit import audit_dataset
from bank_marketing.schemas import (
    DATASET_SHA256, DICTIONARY_SHA256, NUMERIC_COLUMNS, OFFICIAL_URL,
    TRAINING_COLUMNS, DomainError,
)
from bank_marketing.storage import atomic_bytes, canonical_bytes, utc_now

MAX_ARCHIVE_BYTES = 100_000_000
MAX_EXPANDED_BYTES = 200_000_000


def _read_bytes(source, max_bytes):
    if isinstance(source, bytes):
        content = source
    elif isinstance(source, (str, Path)):
        try:
            with Path(source).open("rb") as stream:
                content = stream.read(max_bytes + 1)
        except OSError as exc:
            raise DomainError("SCHEMA_INVALID", "Input file cannot be read.") from exc
    else:
        raise DomainError("SCHEMA_INVALID", "Input must be bytes or a local file path.")
    if len(content) > max_bytes:
        raise DomainError("SCHEMA_INVALID", "Input exceeds byte limit.", {"max_bytes": max_bytes})
    return content


def read_csv(source, max_bytes=20000000, max_rows=50000) -> pd.DataFrame:
    if type(max_bytes) is not int or max_bytes < 1 or type(max_rows) is not int or max_rows < 1:
        raise DomainError("SCHEMA_INVALID", "CSV limits must be positive integers.")
    content = _read_bytes(source, max_bytes)
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DomainError("SCHEMA_INVALID", "CSV must be UTF-8 or UTF-8-SIG.") from exc
    if not text or "\x00" in text:
        raise DomainError("SCHEMA_INVALID", "CSV is empty or contains null bytes.")
    candidates = []
    for delimiter in (";", ","):
        try:
            header = next(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True))
        except (csv.Error, StopIteration):
            continue
        if len(header) > 1:
            candidates.append((delimiter, header))
    if len(candidates) != 1:
        raise DomainError("SCHEMA_INVALID", "CSV header must unambiguously use semicolon or comma.")
    delimiter, header = candidates[0]
    if any(not name or name != name.strip() for name in header) or len(header) != len(set(header)):
        raise DomainError("SCHEMA_INVALID", "CSV headers must be nonblank, exact and unique before pandas parsing.")
    rows = 0
    try:
        parser = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
        next(parser)
        for input_row_id, fields in enumerate(parser):
            if len(fields) != len(header):
                raise DomainError("SCHEMA_INVALID", "CSV row field count differs from header.",
                                  {"input_row_id": input_row_id, "expected": len(header), "actual": len(fields)})
            rows += 1
            if rows > max_rows:
                raise DomainError("SCHEMA_INVALID", "CSV exceeds row limit.", {"max_rows": max_rows})
    except csv.Error as exc:
        raise DomainError("SCHEMA_INVALID", "CSV quoting is malformed.") from exc
    if rows == 0:
        raise DomainError("SCHEMA_INVALID", "CSV must contain data rows.")
    try:
        frame = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, keep_default_na=False,
                            na_filter=False, skip_blank_lines=False, on_bad_lines="error")
    except (pd.errors.ParserError, ValueError) as exc:
        raise DomainError("SCHEMA_INVALID", "CSV parsing failed.") from exc
    if list(frame.columns) != header or len(frame) != rows:
        raise DomainError("SCHEMA_INVALID", "CSV parser did not preserve rows and headers.")
    for column in NUMERIC_COLUMNS:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            valid = values.notna() & frame[column].eq(frame[column].str.strip())
            frame[column] = frame[column].astype(object)
            frame.loc[valid, column] = values.loc[valid]
    frame.attrs.update({"separator": delimiter, "encoding": "utf-8-sig", "input_row_id": "zero_based_original_position"})
    return frame


def _validate_zip_member(info):
    name = info.filename.replace("\\", "/")
    parts = PurePosixPath(name).parts
    unix_mode = info.external_attr >> 16
    if (name.startswith("/") or re.match(r"^[A-Za-z]:", name) or ".." in parts
            or any(":" in part for part in parts) or "\x00" in name
            or stat.S_ISLNK(unix_mode) or info.flag_bits & 1):
        raise DomainError("SCHEMA_INVALID", "Unsafe or encrypted ZIP entry rejected.")
    if info.file_size > MAX_EXPANDED_BYTES:
        raise DomainError("SCHEMA_INVALID", "ZIP entry exceeds extraction limit.")
    return name


def _archive_files(content, depth=0, budget=None):
    if depth > 2:
        raise DomainError("SCHEMA_INVALID", "Nested ZIP depth exceeds supported official layout.")
    budget = [0] if budget is None else budget
    selected = {}
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            seen = set()
            for info in archive.infolist():
                name = _validate_zip_member(info)
                if name in seen:
                    raise DomainError("SCHEMA_INVALID", "Duplicate ZIP entry rejected.")
                seen.add(name)
                budget[0] += info.file_size
                if budget[0] > MAX_EXPANDED_BYTES:
                    raise DomainError("SCHEMA_INVALID", "ZIP expanded size exceeds limit.")
                if info.is_dir():
                    continue
                base = PurePosixPath(name).name
                if base in {"bank-additional-full.csv", "bank-additional-names.txt"}:
                    if base in selected:
                        raise DomainError("SCHEMA_INVALID", "Ambiguous dataset entries in ZIP.")
                    selected[base] = archive.read(info)
                elif base == "bank-additional.zip":
                    nested = _archive_files(archive.read(info), depth + 1, budget)
                    if set(nested) & set(selected):
                        raise DomainError("SCHEMA_INVALID", "Ambiguous nested dataset entries.")
                    selected.update(nested)
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise DomainError("SCHEMA_INVALID", "Official archive cannot be safely read.") from exc
    return selected


def _publish_immutable(path, content):
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise DomainError("DATASET_HASH_MISMATCH", "Existing immutable dataset file differs.")
        return
    try:
        atomic_bytes(path, content, exclusive=True)
    except FileExistsError:
        if path.read_bytes() != content:
            raise DomainError("DATASET_HASH_MISMATCH", "Concurrent immutable publication differs.")


def ingest_dataset(source=None, destination="data/raw") -> dict:
    destination = Path(destination)
    names = ("bank-additional-full.csv", "bank-additional-names.txt")
    if source is None and all((destination / name).is_file() for name in names):
        source = destination
    if source is None or source == OFFICIAL_URL:
        try:
            with urlopen(OFFICIAL_URL, timeout=60) as response:
                if response.geturl() != OFFICIAL_URL:
                    raise DomainError("SCHEMA_INVALID", "Official archive redirect rejected.")
                content = response.read(MAX_ARCHIVE_BYTES + 1)
        except OSError as exc:
            raise DomainError("SCHEMA_INVALID", "Official UCI archive download failed.") from exc
        if len(content) > MAX_ARCHIVE_BYTES:
            raise DomainError("SCHEMA_INVALID", "Official archive exceeds byte limit.")
        files = _archive_files(content)
        source_kind = "official_https"
    else:
        if isinstance(source, str) and "://" in source:
            raise DomainError("SCHEMA_INVALID", "Only the exact official UCI HTTPS URL is allowed.")
        path = Path(source)
        source_kind = "verified_local_copy"
        if path.is_dir():
            files = {name: _read_bytes(path / name, 20000000) for name in names}
        elif path.suffix.lower() == ".csv":
            if path.name != names[0]:
                raise DomainError("DATASET_VARIANT_MISMATCH", "Expected bank-additional-full.csv.")
            files = {names[0]: _read_bytes(path, 20000000), names[1]: _read_bytes(path.with_name(names[1]), 20000000)}
        else:
            files = _archive_files(_read_bytes(path, MAX_ARCHIVE_BYTES))
    if set(files) != set(names):
        raise DomainError("DATASET_VARIANT_MISMATCH", "Archive must contain full additional dataset and its dictionary.")
    for name, expected in zip(names, (DATASET_SHA256, DICTIONARY_SHA256)):
        if hashlib.sha256(files[name]).hexdigest() != expected:
            raise DomainError("DATASET_HASH_MISMATCH", "Official dataset or dictionary checksum differs.", {"file": name, "expected": expected})
    frame = read_csv(files[names[0]])
    report = audit_dataset(frame)
    if not report["valid"]:
        raise DomainError("SCHEMA_INVALID", "Official dataset audit failed.", report["errors"])
    if len(frame) != 41188 or tuple(frame.columns) != TRAINING_COLUMNS:
        raise DomainError("DATASET_VARIANT_MISMATCH", "Official additional-full shape or column order differs.")
    manifest_path = destination.parent / "manifests" / "dataset_manifest.json"
    for name in names:
        path = destination / name
        if path.exists() and (not path.is_file() or path.read_bytes() != files[name]):
            raise DomainError("DATASET_HASH_MISMATCH", "Existing raw file differs; overwrite forbidden.")
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DomainError("DATASET_HASH_MISMATCH", "Existing immutable manifest is invalid.") from exc
        if (not isinstance(existing, dict) or existing.get("dataset_sha256") != DATASET_SHA256 or existing.get("dictionary_sha256") != DICTIONARY_SHA256
                or existing.get("rows") != 41188 or existing.get("schema") != list(TRAINING_COLUMNS)
                or existing.get("files") != {name: {"sha256": hashlib.sha256(files[name]).hexdigest(), "size_bytes": len(files[name])} for name in names}):
            raise DomainError("DATASET_HASH_MISMATCH", "Existing immutable manifest differs from verified source.")
        for name in names:
            _publish_immutable(destination / name, files[name])
        return existing
    manifest = {
        "schema_version": "1.0", "dataset_version": "bank-additional-full_" + DATASET_SHA256[:12],
        "variant": "bank-additional-full", "source_url": OFFICIAL_URL, "source_kind": source_kind,
        "created_at_utc": utc_now(), "dataset_sha256": DATASET_SHA256, "sha256": DATASET_SHA256,
        "dictionary_sha256": DICTIONARY_SHA256, "rows": len(frame), "columns": len(frame.columns),
        "schema": list(frame.columns), "size_bytes": len(files[names[0]]),
        "row_id": {"basis": "original CSV data position", "start": 0, "end_exclusive": len(frame)},
        "files": {name: {"sha256": hashlib.sha256(files[name]).hexdigest(), "size_bytes": len(files[name])} for name in names},
        "license": "CC BY 4.0", "citation": "Moro, S., Cortez, P., and Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems. DOI: 10.1016/j.dss.2014.03.001",
        "audit_counts": report["counts"],
    }
    for name in names:
        _publish_immutable(destination / name, files[name])
    _publish_immutable(manifest_path, canonical_bytes(manifest))
    return manifest

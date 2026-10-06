from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import uuid

import pandas as pd

from bank_marketing.config import load_config
from bank_marketing.data.audit import audit_dataset
from bank_marketing.data.ingest import ingest_dataset, read_csv
from bank_marketing.data.splits import create_splits
from bank_marketing.ml.calibrate import calibrate_models
from bank_marketing.ml.final import evaluate_final
from bank_marketing.ml.registry import Registry
from bank_marketing.ml.selection import freeze_selection, select_model
from bank_marketing.ml.train import train_models
from bank_marketing.schemas import DomainError
from bank_marketing.services.export import export_errors, export_metadata, export_results
from bank_marketing.services.ranking import rank_records
from bank_marketing.services.scoring import score_records
from bank_marketing.storage import atomic_bytes, file_hash, read_json, write_json


def parser():
    root = argparse.ArgumentParser(prog="bank-marketing", description="DepoRank: runner offline kampanye deposito")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("ingest", "audit", "split", "train"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--config", default="configs/experiment.yaml")
        if name == "ingest":
            cmd.add_argument("--source", default=None)
        if name == "train":
            cmd.add_argument("--run-id", default=None)
    for name in ("calibrate", "select", "freeze", "test", "activate", "verify", "report", "benchmark", "interpret"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--run-id", required=True)
    score = commands.add_parser("score")
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    capacity = score.add_mutually_exclusive_group()
    capacity.add_argument("--k", type=int)
    capacity.add_argument("--q", type=float)
    score.add_argument("--demo-labeled", action="store_true")
    score.add_argument("--top-k-only", action="store_true")
    score.add_argument("--errors-output")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
        if result is not None:
            print(json.dumps(result, ensure_ascii=True, allow_nan=False, default=str), flush=True)
        return 0
    except DomainError as error:
        print(json.dumps({"code": error.code, "message": error.message, "error_id": error.error_id}, ensure_ascii=True), file=sys.stderr)
        if error.code in ("ARTIFACT_INTEGRITY_ERROR", "ENVIRONMENT_MISMATCH", "MODEL_NOT_READY"):
            return 3
        return 2
    except (ImportError, FileNotFoundError) as error:
        print(json.dumps({"code": "MODEL_NOT_READY", "message": "Dependency atau berkas wajib belum tersedia.", "error_id": uuid.uuid4().hex}), file=sys.stderr)
        logging.error("CLI dependency failure: %s", type(error).__name__)
        return 3
    except Exception as error:
        error_id = uuid.uuid4().hex
        logging.exception("CLI internal error_id=%s", error_id)
        print(json.dumps({"code": "INTERNAL_ERROR", "message": "Proses gagal; periksa log internal.", "error_id": error_id}), file=sys.stderr)
        return 1


def _score_paths(args):
    output = Path(args.output).resolve()
    if output.suffix.lower() not in (".csv", ".json"):
        raise DomainError("SCHEMA_INVALID", "Output scoring harus .csv atau .json.")
    paths = [output, output.with_suffix(".metadata.json")]
    if args.errors_output:
        paths.append(Path(args.errors_output).resolve())
    source = Path(args.input).resolve()
    protected = [Path("data/raw").resolve(), Path("data/manifests").resolve(), Path("artifacts/runs").resolve()]
    for index, path in enumerate(paths):
        if (path == source or (path.exists() and source.exists() and os.path.samefile(path, source))
                or any(path.is_relative_to(root) for root in protected)
                or path == Path("artifacts/active_run.json").resolve()
                or any(path == other or (path.exists() and other.exists() and os.path.samefile(path, other)) for other in paths[:index])):
            raise DomainError("SCHEMA_INVALID", "Output tidak boleh menimpa input, raw, manifest atau artefak model.")
    return output, paths[1]
def dispatch(args):
    if args.command in ("ingest", "audit", "split", "train"):
        config = load_config(args.config)
        if args.command == "ingest":
            return ingest_dataset(args.source, destination=str(Path(config["dataset"]["path"]).parent))
        path = Path(config["dataset"]["path"])
        if file_hash(path) != config["dataset"]["sha256"]:
            raise DomainError("DATASET_HASH_MISMATCH", "Checksum raw tidak sesuai konfigurasi.")
        data = pd.read_csv(path, sep=config["dataset"]["separator"], encoding="utf-8-sig")
        audit = audit_dataset(data, mode="training")
        if not audit["valid"]:
            raise DomainError("SCHEMA_INVALID", "Audit dataset gagal.", audit["errors"])
        if args.command == "audit":
            write_json("data/manifests/audit.json", audit)
            return {"valid": True, "audit": "data/manifests/audit.json", "counts": audit["counts"]}
        if args.command == "split":
            splits = create_splits(data, config)
            write_json("data/manifests/split_manifest.json", splits)
            return splits
        return {"run_id": train_models(data, config, args.run_id)}
    if args.command == "calibrate":
        return {"run_id": calibrate_models(args.run_id), "status": "calibrated"}
    if args.command == "select":
        selection = select_model(args.run_id)
        return {key: selection[key] for key in ("winner_model_id", "eligible_candidates", "reason")}
    if args.command == "freeze":
        return freeze_selection(args.run_id)
    if args.command == "test":
        return evaluate_final(args.run_id)
    if args.command == "activate":
        ref = Registry().activate(args.run_id)
        return {"run_id": ref.run_id, "winner_model_id": ref.model_id, "status": "active"}
    if args.command == "verify":
        return Registry().verify_run(args.run_id)
    if args.command == "score":
        output, metadata_path = _score_paths(args)
        ref = Registry().active()
        records = read_csv(Path(args.input))
        try:
            scores = score_records(records, ref, mode="demo" if args.demo_labeled else "scoring")
        except DomainError as error:
            if args.errors_output and error.details:
                errors = error.details.get("errors", []) if isinstance(error.details, dict) else error.details
                atomic_bytes(args.errors_output, export_errors(errors))
            raise
        capacity = {"k": args.k} if args.k is not None else {"q": args.q if args.q is not None else 0.1}
        ranked = rank_records(scores, capacity)
        result_bytes = export_results(ranked, format=output.suffix.lower()[1:], top_k_only=args.top_k_only)
        metadata_bytes = export_metadata(ranked)
        atomic_bytes(output, result_bytes)
        atomic_bytes(metadata_path, metadata_bytes)
        return {"output": args.output, "capacity_k": ranked["capacity_k"], "call_fraction": ranked["call_fraction"], "rows": len(ranked["rows"])}
    if args.command == "report":
        from bank_marketing.reporting import generate_reports
        return generate_reports(args.run_id)
    if args.command == "benchmark":
        from bank_marketing.benchmark import benchmark_run
        return benchmark_run(args.run_id)
    if args.command == "interpret":
        from bank_marketing.ml.interpretation import interpret_selection
        result = interpret_selection(args.run_id)
        return {"run_id": args.run_id, "split": "selection", "features": len(result["permutation_importance"]), "subgroups": len(result["subgroups"])}
    raise DomainError("SCHEMA_INVALID", "Command tidak dikenal.")


if __name__ == "__main__":
    sys.exit(main())

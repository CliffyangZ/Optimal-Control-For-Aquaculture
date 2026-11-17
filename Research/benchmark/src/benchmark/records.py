"""Hashing and run-record I/O (contracts/run-record.schema.json)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import jsonschema

from .config import REPO_ROOT

SCHEMA_PATH = (
    REPO_ROOT / "specs" / "006-model-comparison-benchmark" / "contracts" / "run-record.schema.json"
)
STATUSES = ("pending", "trained", "evaluated", "failed")
_ALLOWED = {
    "pending": {"pending", "trained", "failed"},
    "trained": {"trained", "evaluated", "failed"},
    "evaluated": {"evaluated", "failed"},
    "failed": {"failed", "pending"},  # a failed run may be retried from scratch
}


class RecordError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def config_hash(config_raw: dict[str, Any]) -> str:
    return sha256_json(config_raw)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_record(rec: dict[str, Any]) -> None:
    try:
        jsonschema.validate(rec, _schema())
    except jsonschema.ValidationError as e:
        raise RecordError(f"invalid run record: {e.message}") from e


def new_record(run_id: str, family: str, size: str, seed: int, **extra: Any) -> dict[str, Any]:
    rec = {"run_id": run_id, "family": family, "size": size, "seed": seed, "status": "pending"}
    rec.update(extra)
    validate_record(rec)
    return rec


def record_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "run-record.json"


def write_record(run_dir: str | Path, rec: dict[str, Any]) -> Path:
    validate_record(rec)
    p = record_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(p)
    return p


def read_record(run_dir: str | Path) -> dict[str, Any]:
    p = record_path(run_dir)
    rec = json.loads(p.read_text(encoding="utf-8"))
    validate_record(rec)
    return rec


def set_status(rec: dict[str, Any], status: str, failure_reason: str | None = None) -> dict[str, Any]:
    if status not in STATUSES:
        raise RecordError(f"unknown status {status}")
    cur = rec["status"]
    if status not in _ALLOWED[cur]:
        raise RecordError(f"illegal status transition {cur} -> {status}")
    out = dict(rec)
    out["status"] = status
    if status == "failed":
        out["failure_reason"] = failure_reason or "unknown"
    else:
        out.pop("failure_reason", None)
    return out

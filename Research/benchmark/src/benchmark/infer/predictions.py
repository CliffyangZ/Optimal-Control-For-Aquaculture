"""Common prediction file (contracts/predictions.schema.json)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from ..config import REPO_ROOT

SCHEMA_PATH = (
    REPO_ROOT / "specs" / "006-model-comparison-benchmark" / "contracts" / "predictions.schema.json"
)
EVAL_SETS = ("train", "val", "heldout_real", "synthetic")


class PredictionError(ValueError):
    pass


def validate_predictions(doc: dict[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    # `train` is allowed by the plan (task T053); accept it even if the schema file lags behind.
    schema["properties"]["eval_set"]["enum"] = list(EVAL_SETS)
    try:
        jsonschema.validate(doc, schema)
    except jsonschema.ValidationError as e:
        raise PredictionError(f"invalid predictions: {e.message}") from e


def make_doc(run_id: str, eval_set: str, conf_floor: float, max_det: int,
             predictions: list[dict[str, Any]]) -> dict[str, Any]:
    doc = {"run_id": run_id, "eval_set": eval_set, "conf_floor": conf_floor,
           "max_det": max_det, "predictions": predictions}
    validate_predictions(doc)
    return doc


def write_predictions(path: str | Path, doc: dict[str, Any]) -> Path:
    validate_predictions(doc)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def read_predictions(path: str | Path) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_predictions(doc)
    return doc


def prediction_path(run_dir: str | Path, eval_set: str) -> Path:
    return Path(run_dir) / f"predictions.{eval_set}.json"

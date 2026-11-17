"""Evaluation table (CSV) built from per-run metrics.json files."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ..config import Config

GROUPS = ("overall", "feed", "fish")
PRF = ("precision", "recall", "f1")


def all_run_ids(cfg: Config) -> list[str]:
    return list(cfg["runs"])


def _fmt(v) -> str:
    return "" if v is None else f"{v:.4f}"


def build_rows(cfg: Config, run_ids: list[str]) -> tuple[list[str], list[list[str]], list[str]]:
    header = ["run"]
    for g in GROUPS:
        header += [f"{g}_ap_50_95", f"{g}_ap_50"]
        for m in PRF:
            header += [f"{g}_{m}", f"{g}_{m}_ci_low", f"{g}_{m}_ci_high"]
    sys_keys = ("latency_ms_p50", "latency_ms_p95", "latency_ms_e2e_p50", "latency_ms_e2e_p95",
                "fps_effective", "vram_peak_mb")
    header += list(sys_keys)
    rows, problems = [], []
    for rid in run_ids:
        mp = cfg.run_dir(rid) / "metrics.json"
        if not mp.exists():
            problems.append(f"{rid}: no metrics.json (not evaluated)")
            continue
        doc = json.loads(mp.read_text())
        det = doc["detection"]
        row = [rid]
        for g in GROUPS:
            d = det[g]
            row += [_fmt(d["ap_50_95"]), _fmt(d["ap_50"])]
            for m in PRF:
                row += [_fmt(d[m]["value"]), _fmt(d[m]["ci_low"]), _fmt(d[m]["ci_high"])]
        sysb = doc.get("system")
        if sysb is None:
            problems.append(f"{rid}: no latency (run `profile`)")
        row += [_fmt(sysb.get(k)) if sysb else "" for k in sys_keys]
        if "" in row[:-len(sys_keys)]:
            problems.append(f"{rid}: empty cells (class without ground truth?)")
        rows.append(row)
    return header, rows, problems


def write_evaluation_table(cfg: Config, run_ids: list[str]) -> tuple[Path, list[str]]:
    header, rows, problems = build_rows(cfg, run_ids)
    out = cfg.results_root / "report" / "evaluation_table.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return out, problems

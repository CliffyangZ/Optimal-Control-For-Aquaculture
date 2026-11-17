"""`python -m benchmark.cli <stage>`: split, train, predict, evaluate, report (table only).

Profile/synth stages are not implemented yet.
Exit codes: 0 ok, 2 invalid config/input, 3 locked artifact, 4 run failed.
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import ConfigError, family_of_run, load_config  # noqa: F401


def _out(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False))


def cmd_split(args, cfg) -> int:
    from .data.split import LockedArtifactError, load_manifest, write_split
    try:
        p = write_split(cfg, force=args.force)
    except LockedArtifactError as e:
        print(str(e), file=sys.stderr)
        return 3
    m = load_manifest(cfg)
    _out({"stage": "split", "manifest": str(p), "split_hash": m["split_hash"],
          "counts": {k: len(v) for k, v in m["partitions"].items()}})
    return 0


def cmd_train(args, cfg) -> int:
    from .train.common import TrainingFailed, run_training

    rc = 0
    for run_id in args.run:
        fam = family_of_run(run_id)
        if fam == "yolo26":
            from .train.yolo import train as fn
        elif fam == "rfdetr":
            from .train.rfdetr import train as fn
        else:
            from .train.dfine import train as fn
        try:
            rec = run_training(cfg, run_id, fn, micro_batch_override=args.micro_batch, force=args.force)
            _out({"stage": "train", "run": run_id, "status": rec["status"],
                  "weights": rec.get("weights_path")})
        except TrainingFailed as e:
            print(str(e), file=sys.stderr)
            _out({"stage": "train", "run": run_id, "status": "failed"})
            rc = 4
        except (FileNotFoundError, FileExistsError, ConfigError) as e:
            print(str(e), file=sys.stderr)
            return 2
    return rc


def cmd_predict(args, cfg) -> int:
    from .infer.common import eval_set_images, get_adapter
    from .infer.predictions import make_doc, prediction_path, write_predictions
    from .records import read_record

    rc = 0
    for run_id in args.run:
        try:
            run_dir = cfg.run_dir(run_id)
            rec = read_record(run_dir)
            if rec["status"] not in ("trained", "evaluated"):
                raise ConfigError(f"{run_id} is '{rec['status']}', not trained")
            out = prediction_path(run_dir, args.eval_set)
            if out.exists() and not args.force:
                raise FileExistsError(f"{out} exists; pass --force to regenerate")
            preds = get_adapter(run_id)(cfg, rec, eval_set_images(cfg, args.eval_set))
            t = cfg.thresholds
            doc = make_doc(run_id, args.eval_set, float(t["ap_conf_floor"]), int(t["max_det"]), preds)
            write_predictions(out, doc)
            _out({"stage": "predict", "run": run_id, "eval_set": args.eval_set,
                  "n_predictions": len(preds), "file": str(out)})
        except (FileNotFoundError, FileExistsError, ConfigError, NotImplementedError) as e:
            print(f"{run_id}: {e}", file=sys.stderr)
            return 2
        except Exception as e:  # noqa: BLE001
            print(f"{run_id}: predict failed: {type(e).__name__}: {e}", file=sys.stderr)
            rc = 4
    return rc


def cmd_evaluate(args, cfg) -> int:
    from .eval.runner import evaluate_run

    rc = 0
    for run_id in args.run:
        try:
            m = evaluate_run(cfg, run_id, args.eval_set)
            d = m["detection"]["overall"]
            _out({"stage": "evaluate", "run": run_id, "eval_set": args.eval_set,
                  "ap_50_95": d["ap_50_95"], "ap_50": d["ap_50"], "f1": d["f1"]["value"]})
        except (FileNotFoundError, ConfigError, NotImplementedError) as e:
            print(f"{run_id}: {e}", file=sys.stderr)
            return 2
    return rc


def cmd_profile(args, cfg) -> int:
    import json as _json
    from .eval.runner import metrics_path
    from .infer.profile import profile_run
    from .records import read_record

    rc = 0
    for run_id in args.run:
        try:
            run_dir = cfg.run_dir(run_id)
            rec = read_record(run_dir)
            mp = metrics_path(run_dir, "heldout_real")
            if not mp.exists():
                raise FileNotFoundError(f"{mp} missing; run evaluate first")
            sysb = profile_run(cfg, rec)
            m = _json.loads(mp.read_text())
            m["system"] = sysb
            mp.write_text(_json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
            _out({"stage": "profile", "run": run_id, "p50_ms": round(sysb["latency_ms_p50"], 2),
                  "e2e_p50_ms": round(sysb["latency_ms_e2e_p50"], 2),
                  "fps": round(sysb["fps_effective"], 1), "vram_mb": round(sysb["vram_peak_mb"])})
        except (FileNotFoundError, NotImplementedError, ConfigError) as e:
            print(f"{run_id}: {e}", file=sys.stderr)
            rc = 2
        except Exception as e:  # noqa: BLE001
            print(f"{run_id}: profile failed: {type(e).__name__}: {e}", file=sys.stderr)
            rc = 4
    return rc


def cmd_report(args, cfg) -> int:
    from .report.tables import all_run_ids, write_evaluation_table

    ids = all_run_ids(cfg) if args.runs == "all" else args.runs.split(",")
    for r in ids:
        cfg.run_spec(r)  # validates the id
    out, problems = write_evaluation_table(cfg, ids)
    for p in problems:
        print(p, file=sys.stderr)
    _out({"stage": "report", "table": str(out), "missing_or_incomplete": problems})
    return 4 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmark")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="stage", required=True)
    sp = sub.add_parser("split"); sp.add_argument("--force", action="store_true")
    tr = sub.add_parser("train")
    tr.add_argument("--run", action="append", required=True, help="e.g. Y-s, R-m, D-n[-seed7]")
    tr.add_argument("--micro-batch", type=int, default=None)
    tr.add_argument("--force", action="store_true")
    pf = sub.add_parser("profile")
    pf.add_argument("--run", action="append", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("--runs", default="all", help="'all' or comma list, e.g. Y-s,R-s,D-s")
    for name, fn in (("predict", None), ("evaluate", None)):
        sp2 = sub.add_parser(name)
        sp2.add_argument("--run", action="append", required=True)
        sp2.add_argument("--eval-set", default="heldout_real",
                         choices=["train", "val", "heldout_real", "synthetic"])
        if name == "predict":
            sp2.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    return {"split": cmd_split, "train": cmd_train, "predict": cmd_predict,
            "evaluate": cmd_evaluate, "report": cmd_report, "profile": cmd_profile}[args.stage](args, cfg)


if __name__ == "__main__":
    sys.exit(main())

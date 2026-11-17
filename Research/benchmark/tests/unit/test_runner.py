import json

from benchmark.config import load_config
from benchmark.eval import runner
from benchmark.infer.adapter_rfdetr import class_names_from_ids
from benchmark.infer.predictions import make_doc, write_predictions


def test_rfdetr_ids_are_zero_based():
    assert class_names_from_ids([0, 1, 2, 3], ["debris", "feed", "fish", "lumpfish"]) == \
        ["debris", "feed", "fish", "lumpfish"]


def test_evaluate_run_is_deterministic(tmp_path, monkeypatch):
    cfg = load_config()
    gt = {"images": [{"id": i, "width": 100, "height": 100, "file_name": str(i)} for i in range(5)],
          "annotations": [{"id": i + 1, "image_id": i, "category_id": 2, "bbox": [5, 5, 20, 20],
                           "area": 400, "iscrowd": 0} for i in range(5)],
          "categories": []}
    preds = [{"image_id": i, "category": "feed", "bbox": [5, 5, 20, 20], "score": 0.9} for i in range(4)]
    monkeypatch.setattr(runner, "eval_set_gt", lambda c, s: gt)
    monkeypatch.setattr(type(cfg), "run_dir", lambda self, rid: tmp_path)
    rec = {"run_id": "Y-s", "family": "yolo26", "size": "s", "seed": 42, "status": "trained"}
    (tmp_path / "run-record.json").write_text(json.dumps(rec))
    write_predictions(tmp_path / "predictions.heldout_real.json",
                      make_doc("Y-s", "heldout_real", 0.001, 300, preds))
    a = runner.evaluate_run(cfg, "Y-s")
    b = runner.evaluate_run(cfg, "Y-s")
    assert a == b
    assert abs(a["detection"]["feed"]["recall"]["value"] - 0.8) < 1e-9
    assert json.loads((tmp_path / "run-record.json").read_text())["status"] == "evaluated"


def test_evaluate_keeps_system_block(tmp_path, monkeypatch):
    cfg = load_config()
    gt = {"images": [{"id": 0, "width": 100, "height": 100, "file_name": "0"}],
          "annotations": [{"id": 1, "image_id": 0, "category_id": 2, "bbox": [5, 5, 20, 20],
                           "area": 400, "iscrowd": 0}], "categories": []}
    monkeypatch.setattr(runner, "eval_set_gt", lambda c, s: gt)
    monkeypatch.setattr(type(cfg), "run_dir", lambda self, rid: tmp_path)
    rec = {"run_id": "Y-s", "family": "yolo26", "size": "s", "seed": 42, "status": "trained"}
    (tmp_path / "run-record.json").write_text(json.dumps(rec))
    write_predictions(tmp_path / "predictions.heldout_real.json", make_doc("Y-s", "heldout_real", 0.001, 300, []))
    (tmp_path / "metrics.json").write_text(json.dumps({"system": {"fps_effective": 123.0}}))
    assert runner.evaluate_run(cfg, "Y-s")["system"] == {"fps_effective": 123.0}

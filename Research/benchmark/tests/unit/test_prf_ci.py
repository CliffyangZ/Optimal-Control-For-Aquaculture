import numpy as np

from benchmark.eval.prf_ci import iou_xywh, match_image, prf_with_ci

CLASSES = ["debris", "feed", "fish", "lumpfish"]


def gt_doc(n=30):
    imgs, anns = [], []
    for i in range(n):
        imgs.append({"id": i, "width": 200, "height": 200, "file_name": str(i)})
        anns.append({"id": 2 * i + 1, "image_id": i, "category_id": 2, "bbox": [10, 10, 20, 20],
                     "area": 400, "iscrowd": 0})
        anns.append({"id": 2 * i + 2, "image_id": i, "category_id": 3, "bbox": [100, 100, 40, 40],
                     "area": 1600, "iscrowd": 0})
    return {"images": imgs, "annotations": anns, "categories": []}


def perfect(n=30):
    out = []
    for i in range(n):
        out.append({"image_id": i, "category": "feed", "bbox": [10, 10, 20, 20], "score": 0.9})
        out.append({"image_id": i, "category": "fish", "bbox": [100, 100, 40, 40], "score": 0.9})
    return out


def run(gt, preds, seed=1, B=200):
    return prf_with_ci(gt, preds, CLASSES, 0.25, 0.5, B, seed, report_classes=["feed", "fish"])


def test_iou_basic():
    assert abs(iou_xywh(np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10]]))[0, 0] - 1) < 1e-9
    assert iou_xywh(np.array([[0, 0, 10, 10]]), np.array([[20, 20, 5, 5]]))[0, 0] == 0


def test_match_is_one_to_one():
    gt = np.array([[0, 0, 10, 10]])
    pb = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
    assert match_image(gt, pb, np.array([0.9, 0.8]), 0.5) == (1, 1, 0)


def test_perfect_gives_one_with_zero_width_ci():
    r = run(gt_doc(), perfect())
    for g in ("overall", "feed", "fish"):
        for m in ("precision", "recall", "f1"):
            d = r[g][m]
            assert d["value"] == d["ci_low"] == d["ci_high"] == 1.0


def test_low_score_predictions_ignored_and_fp_fn_counted():
    preds = perfect()
    preds = [p for p in preds if not (p["category"] == "feed" and p["image_id"] < 10)]  # 10 misses
    preds += [{"image_id": 0, "category": "fish", "bbox": [0, 150, 20, 20], "score": 0.8}]  # 1 FP
    preds += [{"image_id": 1, "category": "fish", "bbox": [0, 150, 20, 20], "score": 0.1}]  # < conf
    r = run(gt_doc(), preds)
    assert abs(r["feed"]["recall"]["value"] - 20 / 30) < 1e-9
    assert r["feed"]["precision"]["value"] == 1.0
    assert abs(r["fish"]["precision"]["value"] - 30 / 31) < 1e-9


def test_seeded_and_ci_contains_point():
    preds = [p for p in perfect() if p["image_id"] % 3]
    a, b = run(gt_doc(), preds, seed=5), run(gt_doc(), preds, seed=5)
    assert a == b
    d = a["feed"]["recall"]
    assert d["ci_low"] <= d["value"] <= d["ci_high"] and d["ci_high"] > d["ci_low"]


def test_resampling_unit_is_image():
    # all errors concentrated in one image => CI must be wider than for errors spread over images
    gt = gt_doc(20)
    one_bad = [p for p in perfect(20) if p["image_id"] != 0]
    spread = [p for p in perfect(20) if not (p["category"] == "feed" and p["image_id"] % 2 == 0)]
    w1 = run(gt, one_bad)["overall"]["recall"]
    assert w1["ci_high"] <= 1.0 and w1["ci_low"] < w1["value"] <= w1["ci_high"]
    assert run(gt, spread)["feed"]["recall"]["value"] == 0.5

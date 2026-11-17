from benchmark.eval.coco_ap import coco_ap

CLASSES = ["debris", "feed", "fish", "lumpfish"]


def gt_doc():
    imgs = [{"id": 0, "width": 200, "height": 200, "file_name": "a"},
            {"id": 1, "width": 200, "height": 200, "file_name": "b"}]
    anns = [
        {"id": 1, "image_id": 0, "category_id": 2, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
        {"id": 2, "image_id": 0, "category_id": 3, "bbox": [100, 100, 60, 60], "area": 3600, "iscrowd": 0},
        {"id": 3, "image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 0},
    ]
    return {"images": imgs, "annotations": anns,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]}


def pred(i, cat, bbox, s=0.9):
    return {"image_id": i, "category": cat, "bbox": bbox, "score": s}


def test_perfect_predictions_ap_one():
    preds = [pred(0, "feed", [10, 10, 20, 20]), pred(0, "fish", [100, 100, 60, 60]),
             pred(1, "feed", [50, 50, 20, 20])]
    r = coco_ap(gt_doc(), preds, CLASSES)
    assert r["overall"]["ap_50_95"] == 1.0 and r["overall"]["ap_50"] == 1.0
    assert r["feed"]["ar_300"] == 1.0
    assert r["debris"]["ap_50"] is None  # no GT for this class


def test_no_predictions_gives_zero_where_gt_exists():
    r = coco_ap(gt_doc(), [], CLASSES)
    assert r["overall"]["ap_50"] == 0.0 and r["feed"]["ap_50"] == 0.0
    assert r["lumpfish"]["ap_50"] is None


def test_half_recall_and_size_buckets():
    # feed GT: two small objects; only one is found -> AP@.5 = 0.5; fish (medium, 32²-96²) found
    preds = [pred(0, "feed", [10, 10, 20, 20]), pred(0, "fish", [100, 100, 60, 60])]
    r = coco_ap(gt_doc(), preds, CLASSES)
    assert abs(r["feed"]["ap_50"] - 0.5) < 0.02
    assert r["feed"]["ap_large"] is None
    assert abs(r["fish"]["ap_medium"] - 1.0) < 1e-9 and r["fish"]["ap_small"] is None


def test_wrong_class_does_not_count():
    r = coco_ap(gt_doc(), [pred(0, "fish", [10, 10, 20, 20])], CLASSES)
    assert r["feed"]["ap_50"] == 0.0


def test_dense_image_is_not_capped_at_100_detections():
    n = 250
    gt = {"images": [{"id": 0, "width": 2000, "height": 2000, "file_name": "a"}],
          "annotations": [{"id": i + 1, "image_id": 0, "category_id": 2,
                           "bbox": [(i % 25) * 70, (i // 25) * 70, 20, 20], "area": 400, "iscrowd": 0}
                          for i in range(n)],
          "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}
    preds = [pred(0, "feed", a["bbox"]) for a in gt["annotations"]]
    r = coco_ap(gt, preds, CLASSES)
    assert abs(r["feed"]["ap_50_95"] - 1.0) < 1e-9 and abs(r["feed"]["ar_300"] - 1.0) < 1e-9

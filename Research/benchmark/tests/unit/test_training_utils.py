import numpy as np
import pytest

from benchmark.config import ConfigError, family_of_run, load_config, base_run_id, run_seed
from benchmark.data.split import group_key, make_split
from benchmark.records import RecordError, new_record, set_status
from benchmark.train.common import micro_batch
from benchmark.train.dfine import clip_xywh, lr_factor, to_normalized_cxcywh


def test_config_loads_and_run_ids():
    cfg = load_config()
    assert set(cfg["runs"]) == {f"{f}-{s}" for f in "YRD" for s in "nsm"}
    assert family_of_run("R-m") == "rfdetr" and base_run_id("Y-s-seed7") == "Y-s"
    assert run_seed("Y-s-seed7", 42) == 7 and run_seed("Y-s", 42) == 42
    with pytest.raises(ConfigError):
        family_of_run("X-s")


def test_micro_batch_keeps_effective_batch():
    cfg = load_config()
    for rid in cfg["runs"]:
        mb, acc = micro_batch(cfg, rid)
        assert mb * acc == cfg["budget"]["effective_batch"]
    assert micro_batch(cfg, "R-m", 2) == (2, 8)


def _entries():
    out = []
    for src, grp, n in (("a", "g1", 30), ("a", "g2", 10), ("b", "g3", 20)):
        out += [{"sha256": f"{src}{grp}{i:03d}", "source_dataset": src, "group": grp} for i in range(n)]
    return out


def test_split_deterministic_disjoint_and_complete():
    frac = {"train": 0.6, "val": 0.2, "heldout_real": 0.2}
    s1, s2 = make_split(_entries(), frac, 42), make_split(_entries(), frac, 42)
    assert s1 == s2
    ids = [e["sha256"] for part in s1.values() for e in part]
    assert len(ids) == len(set(ids)) == 60
    assert all(len(s1[p]) > 0 for p in s1)
    assert make_split(_entries(), frac, 43) != s1


def test_group_key():
    assert group_key("downCamera_1-mp4_frame_112_jpg.rf.abc.jpg") == "downCamera_1-mp4"
    assert group_key("Timeline-2_mov-0002_jpg.rf.abc.jpg") == "Timeline-2_mov"


def test_record_transitions():
    r = new_record("Y-s", "yolo26", "s", 42)
    r = set_status(r, "trained")
    with pytest.raises(RecordError):
        set_status(r, "pending")
    f = set_status(r, "failed", "boom")
    assert f["failure_reason"] == "boom"


def test_dfine_helpers():
    assert clip_xywh(-5, -5, 10, 10, 100, 100) == (0.0, 0.0, 5.0, 5.0)
    assert clip_xywh(99.5, 10, 5, 5, 100, 100) is None
    b = to_normalized_cxcywh(np.array([[0, 0, 50, 100]]), 100, 200)
    assert np.allclose(b, [[0.25, 0.25, 0.5, 0.5]])
    assert lr_factor(0, 1000) < lr_factor(99, 1000) <= 1.0
    assert lr_factor(999, 1000) < 0.2

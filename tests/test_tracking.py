from __future__ import annotations

import pytest

from khunghinh.detection.base import FaceBox
from khunghinh.tracking.iou_tracker import IouTracker, iou


def fb(x, y, w, h, score=0.9):
    return FaceBox(x, y, w, h, score)


def test_iou_identical():
    assert iou(fb(0, 0, 10, 10), fb(0, 0, 10, 10)) == 1.0


def test_iou_disjoint():
    assert iou(fb(0, 0, 10, 10), fb(100, 100, 10, 10)) == 0.0


def test_iou_half_overlap():
    assert iou(fb(0, 0, 10, 10), fb(5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_iou_zero_area():
    assert iou(fb(0, 0, 0, 10), fb(0, 0, 10, 10)) == 0.0


def test_min_hits_then_stable_id():
    tr = IouTracker(min_hits=3)
    box = fb(100, 100, 50, 50)
    assert tr.update([box]) == []          # frame 1 (tentative)
    assert tr.update([box]) == []          # frame 2
    out3 = tr.update([box])                 # frame 3 -> confirmed
    assert len(out3) == 1 and out3[0].track_id == 1
    out4 = tr.update([box])
    assert out4[0].track_id == 1


def test_two_boxes_two_ids():
    tr = IouTracker(min_hits=1)
    a, b = fb(0, 0, 40, 40), fb(300, 300, 40, 40)
    ids = set()
    for _ in range(4):
        for o in tr.update([a, b]):
            ids.add(o.track_id)
    assert ids == {1, 2}


def test_low_score_never_spawns():
    tr = IouTracker(min_hits=1, high_score_thresh=0.6)
    low = fb(10, 10, 40, 40, score=0.2)
    for _ in range(5):
        tr.update([low])
    assert tr.tracks == []


def test_death_after_max_age_and_no_id_reuse():
    tr = IouTracker(min_hits=1, max_age=3)
    box = fb(50, 50, 40, 40)
    for _ in range(3):
        tr.update([box])           # confirm track id=1
    for _ in range(4):
        tr.update([])              # miss > max_age -> dead
    assert tr.tracks == []
    out = None
    for _ in range(3):
        out = tr.update([fb(50, 50, 40, 40)])
    assert out and out[0].track_id > 1  # new id, không tái dùng


def test_short_dropout_keeps_id():
    tr = IouTracker(min_hits=1, max_age=30)
    box = fb(50, 50, 40, 40)
    for _ in range(3):
        tr.update([box])
    tr.update([])
    tr.update([])
    out = tr.update([box])
    assert out and out[0].track_id == 1


def test_confirmed_rescued_by_low_score_box():
    tr = IouTracker(min_hits=2, high_score_thresh=0.6, match_low_iou_threshold=0.2)
    box = fb(50, 50, 40, 40, score=0.9)
    for _ in range(3):
        tr.update([box])  # confirmed id 1
    out = tr.update([fb(52, 52, 40, 40, score=0.2)])  # low score, overlapping
    assert out and out[0].track_id == 1


def test_invalid_params_raise():
    for kw in (dict(iou_threshold=0.0), dict(max_age=0), dict(min_hits=0),
               dict(low_score_thresh=0.9, high_score_thresh=0.5)):
        with pytest.raises(ValueError):
            IouTracker(**kw)


def test_output_objects_are_new_input_unmodified():
    tr = IouTracker(min_hits=1)
    box = fb(10, 10, 40, 40)
    tr.update([box])
    out = tr.update([box])
    assert out[0] is not box
    assert box.track_id is None       # input frozen, không bị sửa
    assert out[0].track_id == 1


def test_below_low_score_thresh_never_rescues_track():
    tr = IouTracker(min_hits=2, high_score_thresh=0.6, low_score_thresh=0.1,
                    match_low_iou_threshold=0.2)
    box = fb(50, 50, 40, 40, score=0.9)
    for _ in range(3):
        tr.update([box])  # confirmed id 1
    # Phát hiện gần như nhiễu (score=0.02 < low_score_thresh=0.1) chồng lấn track —
    # KHÔNG được phép "giải cứu" track bằng detection dưới sàn low_score_thresh.
    out = tr.update([fb(52, 52, 40, 40, score=0.02)])
    assert out == []
    assert tr.tracks[0].time_since_update == 1  # track bị miss, không bị ghi đè box


def test_determinism_across_two_trackers():
    seq = [[fb(i * 5, 10, 40, 40), fb(300 - i * 5, 200, 40, 40)] for i in range(6)]
    t1 = IouTracker(min_hits=1)
    t2 = IouTracker(min_hits=1)
    r1 = [[(o.track_id, o.x) for o in t1.update(f)] for f in seq]
    r2 = [[(o.track_id, o.x) for o in t2.update(f)] for f in seq]
    assert r1 == r2

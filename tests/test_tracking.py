import numpy as np
import supervision as sv

from football_analytics.config import PLAYER_CLASS_ID, REFEREE_CLASS_ID
from football_analytics.tracking import BallTracker, PeopleTracker


def ball(*centers, conf=None):
    xyxy = np.array([[x - 5, y - 5, x + 5, y + 5] for x, y in centers], dtype=float)
    conf = np.array(conf if conf is not None else [0.9] * len(centers), dtype=float)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=np.zeros(len(centers), dtype=int))


def test_people_tracker_keeps_ids_and_classes():
    tracker = PeopleTracker(frame_rate=25)
    ids = None
    for step in range(5):
        dets = sv.Detections(
            xyxy=np.array([[100 + step, 100, 140 + step, 200], [500, 100, 540, 200]], dtype=float),
            confidence=np.array([0.9, 0.8]),
            class_id=np.array([PLAYER_CLASS_ID, REFEREE_CLASS_ID]),
        )
        tracked = tracker.update(dets)
        if step == 0:
            # ByteTrack confirms new tracks on their second sighting.
            assert len(tracked) == 0
            continue
        assert len(tracked) == 2
        order = np.argsort(tracked.xyxy[:, 0])
        current = tracked.tracker_id[order].tolist()
        assert tracked.class_id[order].tolist() == [PLAYER_CLASS_ID, REFEREE_CLASS_ID]
        if ids is not None:
            assert current == ids
        ids = current


def test_ball_tracker_rejects_far_outlier_and_bridges_gaps():
    tracker = BallTracker((1000, 1000), max_missing=2, max_jump_ratio=0.1)
    assert len(tracker.update(ball((100, 100)))) == 1
    # A higher-confidence false positive far away loses to the nearby candidate.
    picked = tracker.update(ball((900, 900), (110, 105), conf=[0.95, 0.5]))
    assert picked.get_anchors_coordinates(sv.Position.CENTER)[0].tolist() == [110, 105]
    # Missing for up to max_missing frames: keep the last position.
    assert len(tracker.update(sv.Detections.empty())) == 1
    assert len(tracker.update(sv.Detections.empty())) == 1
    assert len(tracker.update(sv.Detections.empty())) == 0
    # After being lost, the most confident candidate is accepted anywhere.
    picked = tracker.update(ball((900, 900), (110, 105), conf=[0.95, 0.5]))
    assert picked.get_anchors_coordinates(sv.Position.CENTER)[0].tolist() == [900, 900]

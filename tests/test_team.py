import numpy as np
import supervision as sv

from football_analytics.config import GOALKEEPER_CLASS_ID, NO_TEAM, PLAYER_CLASS_ID, REFEREE_CLASS_ID
from football_analytics.team import TeamAssigner, TeamClassifier, extract_features, kit_feature

GRASS = (40, 140, 40)  # BGR
RED = (30, 30, 200)
WHITE = (245, 245, 245)


def make_scene(players):
    """players: list of (x, colour). Returns a frame and detections."""
    frame = np.full((400, 800, 3), GRASS, dtype=np.uint8)
    boxes = []
    for x, colour in players:
        x1, y1, x2, y2 = x, 100, x + 40, 200
        frame[y1:y2, x1:x2] = colour
        boxes.append([x1, y1, x2, y2])
    return frame, np.array(boxes, dtype=float)


def test_kit_feature_ignores_grass_border():
    crop = np.full((20, 20, 3), GRASS, dtype=np.uint8)
    crop[5:15, 5:15] = RED
    feature = kit_feature(crop)
    pure = kit_feature(np.full((10, 10, 3), RED, dtype=np.uint8))
    assert np.allclose(feature, pure, atol=2)


def test_classifier_separates_two_kits_lighter_first():
    frame, boxes = make_scene([(50, RED), (150, WHITE), (250, RED), (350, WHITE), (450, RED), (550, WHITE)])
    dets = sv.Detections(xyxy=boxes, class_id=np.full(len(boxes), PLAYER_CLASS_ID))
    features, valid = extract_features(frame, dets)
    assert len(valid) == 6
    clf = TeamClassifier().fit(features)
    labels = clf.predict(features)
    assert list(labels) == [1, 0, 1, 0, 1, 0]  # white (lighter) is team 0
    b, g, r = clf.team_colors_bgr[1]
    assert r > 150 and g < 80 and b < 80


def test_assigner_votes_goalkeeper_and_referee():
    players = [(50, RED), (100, RED), (600, WHITE), (650, WHITE), (20, (10, 200, 230)), (400, (0, 0, 0))]
    frame, boxes = make_scene(players)
    class_ids = [PLAYER_CLASS_ID] * 4 + [GOALKEEPER_CLASS_ID, REFEREE_CLASS_ID]
    dets = sv.Detections(
        xyxy=boxes, class_id=np.array(class_ids), tracker_id=np.arange(1, len(boxes) + 1)
    )
    assigner = TeamAssigner()
    features, _ = extract_features(frame, dets[dets.class_id == PLAYER_CLASS_ID])
    assert assigner.calibrate(features)
    teams = assigner.assign(frame, dets)
    assert teams[0] == teams[1] == 1  # red
    assert teams[2] == teams[3] == 0  # white
    assert teams[4] == 1  # keeper near the red players
    assert teams[5] == NO_TEAM  # referee


def test_assigner_fits_lazily_without_calibration():
    frame, boxes = make_scene([(50, RED), (150, WHITE), (250, RED), (350, WHITE)])
    dets = sv.Detections(
        xyxy=boxes, class_id=np.full(4, PLAYER_CLASS_ID), tracker_id=np.arange(4)
    )
    assigner = TeamAssigner(min_samples=8)
    assert (assigner.assign(frame, dets) == NO_TEAM).all()
    teams = assigner.assign(frame, dets)  # 8 samples collected -> fitted
    assert assigner.ready
    assert teams[0] == teams[2] != teams[1] == teams[3]

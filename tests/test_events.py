import numpy as np
import supervision as sv

from football_analytics.config import PLAYER_CLASS_ID, Settings
from football_analytics.events import MatchStats, distance_to_boxes, find_ball_owner

# Three players, 100 px tall: ids 1 and 2 on team 0, id 3 on team 1.
PEOPLE = sv.Detections(
    xyxy=np.array([[100, 100, 140, 200], [400, 100, 440, 200], [700, 100, 740, 200]], dtype=float),
    class_id=np.full(3, PLAYER_CLASS_ID),
    tracker_id=np.array([1, 2, 3]),
)
TEAMS = np.array([0, 0, 1])


def ball_at(x, y):
    return sv.Detections(xyxy=np.array([[x - 5, y - 5, x + 5, y + 5]], dtype=float), class_id=np.array([0]))


def test_distance_to_boxes():
    boxes = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=float)
    assert np.allclose(distance_to_boxes(np.array([5, 5]), boxes), [0, 15])
    assert np.allclose(distance_to_boxes(np.array([13, 14]), boxes), [5, np.hypot(7, 4)])


def test_owner_needs_ball_within_relative_distance():
    assert find_ball_owner(PEOPLE, TEAMS, ball_at(120, 195), 0.5).tracker_id == 1
    assert find_ball_owner(PEOPLE, TEAMS, ball_at(120, 290), 0.5) is None  # 85 px > 50 px
    assert find_ball_owner(PEOPLE, TEAMS, sv.Detections.empty(), 0.5) is None


def test_pass_turnover_and_possession():
    stats = MatchStats(fps=25, settings=Settings(possession_confirm_frames=2))
    frame = 0

    def hold(x, frames):
        nonlocal frame
        events = []
        for _ in range(frames):
            event = stats.update(frame, PEOPLE, TEAMS, ball_at(x, 195), weight=1)
            if event:
                events.append(event)
            frame += 1
        return events

    assert hold(120, 10) == []  # player 1 gets the ball
    (event,) = hold(420, 10)  # pass to player 2 (same team)
    assert (event.kind, event.from_player, event.to_player, event.team) == ("pass", 1, 2, 0)
    (event,) = hold(720, 10)  # player 3 intercepts
    assert (event.kind, event.team) == ("turnover", 1)

    assert stats.passes == [1, 0]
    assert stats.turnovers_won == [0, 1]
    # 1 unconfirmed frame at the start, then team 0 holds until frame 21.
    assert stats.possession_frames.tolist() == [20, 9]
    assert round(sum(stats.possession_pct)) == 100


def test_ball_flying_past_is_not_possession():
    stats = MatchStats(fps=25, settings=Settings(possession_confirm_frames=3))
    stats.update(0, PEOPLE, TEAMS, ball_at(120, 195))
    stats.update(1, PEOPLE, TEAMS, ball_at(250, 195))
    assert stats.owner is None
    assert stats.possession_frames.sum() == 0

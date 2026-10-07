"""The harness must be sound before a single token is spent on it."""

import json

from catanatron.models.player import Color, RandomPlayer

from arena.colors import FULL_TABLE
from arena.events import read_run
from arena.runner import bot_seat, run_match


def test_random_games_finish_and_log_cleanly(tmp_path):
    for i in range(12):
        summary = run_match(
            seats=[(c, bot_seat(RandomPlayer)) for c in list(FULL_TABLE)[:3]],
            runs_dir=tmp_path,
            seed=1000 + i,
        )
        events = read_run(summary["path"])
        assert events[0]["kind"] == "game_start"
        assert events[-1]["kind"] == "game_over"
        assert [e["seq"] for e in events] == list(range(len(events)))
        assert summary["turns"] > 0
        # a finished game has a winner; a game that hit the turn limit does not,
        # and either is a legitimate outcome — a crash is not
        assert summary["winner"] in {c.value for c in Color} | {None}


def test_game_start_carries_board_geometry(tmp_path):
    summary = run_match(
        seats=[(c, bot_seat(RandomPlayer)) for c in list(FULL_TABLE)[:2]],
        runs_dir=tmp_path,
        seed=7,
    )
    start = read_run(summary["path"])[0]
    board = start["board"]
    land = [t for t in board["tiles"] if t["tile"].get("type") == "RESOURCE_TILE"]
    desert = [t for t in board["tiles"] if t["tile"].get("type") == "DESERT"]
    assert len(land) + len(desert) == 19
    # geometry walks every tile, water and ports included, so node ids run past
    # the 54 of the land board
    assert len(board["nodes"]) >= 54
    assert board["edges"]
    # the payload must survive a round trip to the browser
    json.dumps(board)

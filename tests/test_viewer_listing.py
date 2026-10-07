"""The run picker: which match is being played now, and which are history.

Choosing a run is the first thing a spectator does. If a dead match is listed as
live, they sit watching a board that will never move; if a live one is not, they
open yesterday's game instead of the one they came for.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import viewer  # noqa: E402


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, "RUNS", tmp_path)
    return tmp_path


def _write(path, *events):
    path.write_text("".join(json.dumps(e) + "\n" for e in events))


START = {"kind": "game_start", "at": 1789683935.0}
STATE = {"kind": "state", "at": 1789683940.0, "blob": "x" * 200_000}
OVER = {"kind": "game_over", "at": 1789685336.0, "winner": "WHITE"}


def _status(runs, name):
    return {r["name"]: r["status"] for r in viewer.list_runs()}[name]


def test_a_match_that_reached_game_over_is_listed_as_finished(runs):
    """Read from the end of the file: a state line far longer than one read block
    sits before `game_over`, which a naive tail would split and misparse."""
    _write(runs / "done.jsonl", START, STATE, STATE, OVER)
    assert _status(runs, "done") == "finished"


def test_a_match_nobody_is_writing_that_never_ended_is_stopped_not_live(runs):
    """A killed process leaves a file that looks exactly like a match in progress.
    Calling it live would leave the spectator waiting on a board that never moves."""
    _write(runs / "dead.jsonl", START, STATE)
    assert _status(runs, "dead") == "stopped"


def test_a_file_held_open_for_writing_is_live(runs):
    """The writer keeps its file open for the whole match, and that open handle is
    what separates a live match from a dead one, not the file's age."""
    path = runs / "playing.jsonl"
    _write(path, START, STATE)
    with path.open("a"):
        assert _status(runs, "playing") == "live"
    assert _status(runs, "playing") == "stopped"


def test_a_file_merely_being_read_is_not_mistaken_for_live(runs):
    """The viewer itself opens run files to replay them. Its own reads must not
    promote a dead match to live."""
    path = runs / "dead.jsonl"
    _write(path, START, STATE)
    with path.open("rb"):
        assert _status(runs, "dead") == "stopped"


def test_live_matches_are_listed_before_newer_finished_ones(runs):
    """The match being played is the one a spectator most likely came for."""
    playing = runs / "playing.jsonl"
    _write(playing, START, STATE)
    _write(runs / "newer.jsonl", START, OVER)
    os.utime(playing, (0, 0))
    with playing.open("a"):
        assert viewer.list_runs()[0]["name"] == "playing"


def test_the_start_time_comes_from_the_match_not_the_filesystem(runs):
    """A copied or touched file changes its mtime; when the match was played does not."""
    _write(runs / "done.jsonl", START, OVER)
    os.utime(runs / "done.jsonl", (0, 0))
    assert viewer.list_runs()[0]["started"] == START["at"]

"""Resuming a match that a crash cut short, from the roll that opens a turn."""

import json
import random

from arena.deciders import ScriptedDecider
from arena.llm_player import LLMPlayer
from arena.personas import ROSTER
from arena.rebuild import resume_point
from arena.runner import llm_seat, run_match
from arena.scripted import heuristic_script
from catanatron.models.player import Color

SEATS = (("GREY", "Plain"), ("RED", "Plain"), ("BLUE", "Plain"), ("BLACK", "Saboteur"))


def seats(seed=0):
    return [(Color[c], llm_seat(ROSTER[p], ScriptedDecider(
        heuristic_script("trader", rng=random.Random(seed + i)), label="dry-run")))
        for i, (c, p) in enumerate(SEATS)]


def torn_copy(path, tmp_path, keep):
    """The first `keep` lines, then what the reboot of 2026-10-08 left: a run of
    zero bytes where the next line should have been."""
    lines = open(path).read().splitlines(keepends=True)[:keep]
    torn = tmp_path / "torn.jsonl"
    torn.write_bytes("".join(lines).encode() + b"\x00" * 4096)
    return torn


def test_a_resumed_match_starts_where_the_torn_log_left_off_and_plays_to_the_end(tmp_path):
    """A reboot cut a saboteur match at turn 75 with three seats one point from
    winning. Replaying it from scratch throws that position away. If the resume
    started anywhere else, or with any seat's hand, buildings or note different
    from the log, the rest of the match would be a different game wearing the
    original's name."""
    original = run_match(seats=seats(), runs_dir=tmp_path / "a", seed=31, max_offers_per_turn=2)
    torn = torn_copy(original["path"], tmp_path, keep=900)

    point = resume_point(torn)
    events = [json.loads(l) for l in open(original["path"])]
    snapshot = next(e for e in events if e["kind"] == "state" and e["seq"] == point.snapshot["seq"])
    assert snapshot["last_action"]["type"] == "ROLL"

    turn = point.game.state.num_turns
    seen = {}
    resumed = run_match(seats=seats(1), runs_dir=tmp_path / "b", resume=point, resume_seed=7,
                        on_event=lambda e: seen.setdefault("first_state", e)
                        if e["kind"] == "state" else None)
    start = json.loads(open(resumed["path"]).readline())
    assert start["resumed_from"] == {"run": original["game_id"], "ply": point.index,
                                     "turn": turn, "resume_seed": 7}
    # the first move played after the resume is the roll the log was cut before
    first = seen["first_state"]
    assert first["last_action"]["type"] == "ROLL"
    assert first["last_action"]["color"] == snapshot["last_action"]["color"]
    assert resumed["winner"] is not None


def test_each_seat_gets_back_the_note_and_history_marker_it_had(tmp_path):
    """The note an agent rereads is its only memory across turns. A resumed seat
    without it would plan from nothing, and its next decisions would not belong
    to the same player."""
    original = run_match(seats=seats(), runs_dir=tmp_path / "a", seed=31, max_offers_per_turn=2)
    point = resume_point(torn_copy(original["path"], tmp_path, keep=900))
    captured = {}

    def grab(event):
        if event["kind"] == "game_start":
            for p in captured["players"]:
                captured[p.color.value] = (p._seen, p._note)

    class Spy(LLMPlayer):
        pass

    made = []

    def factory(color, persona, i):
        def build(c, talk, log):
            p = Spy(c, persona, ScriptedDecider(
                heuristic_script("trader", rng=random.Random(i)), label="dry-run"), talk, log)
            made.append(p)
            return p
        return build

    captured["players"] = made
    run_match(seats=[(Color[c], factory(c, ROSTER[p], i)) for i, (c, p) in enumerate(SEATS)],
              runs_dir=tmp_path / "b", resume=point, resume_seed=3, on_event=grab, max_turns=0)
    assert any(note for _, note in (captured[c] for c, _ in SEATS))
    for color, _ in SEATS:
        assert captured[color] == (point.seen_all.get(color, 0), point.notes_all.get(color))

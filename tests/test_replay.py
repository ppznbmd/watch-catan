"""The replay reader: what a spectator is shown must be what happened.

A viewer that renders the wrong hand is worse than one that renders nothing —
the board still looks like a real game, so nobody checks.
"""

import json
import random

import pytest
from catanatron.models.player import Color
from catanatron.players.value import ValueFunctionPlayer

from arena.deciders import ScriptedDecider
from arena.events import read_run
from arena.personas import ROSTER
from arena.replay import Replay, ReplayError, load
from arena.runner import bot_seat, llm_seat, run_match
from arena.scripted import heuristic_script


def _scripted(seed):
    return ScriptedDecider(heuristic_script("trader", rng=random.Random(seed)),
                           label="dry-run")


@pytest.fixture
def match(tmp_path):
    """One scripted match on disk. Spends nothing."""
    summary = run_match(
        seats=[
            (Color.RED, llm_seat(ROSTER["Hard bargainer"], _scripted(1))),
            (Color.BLUE, llm_seat(ROSTER["Cooperator"], _scripted(2))),
            (Color.ORANGE, bot_seat(ValueFunctionPlayer)),
        ],
        runs_dir=tmp_path, seed=7, max_offers_per_turn=3,
    )
    return summary


def test_a_hand_is_attributed_by_the_shuffled_turn_order(match):
    """`player_state` is keyed P0..P3 by seating index and `State.__init__`
    shuffles the seats. Read in construction order instead, and every hand, every
    victory point and every card count is shown against the wrong player — on a
    board that looks perfectly normal."""
    replay = load(match["path"])
    final = replay.frames[-1]
    for color, agent in match["agents"].items():
        assert final["players"][color]["victory_points"] == agent["victory_points"]


def test_the_opening_frame_already_has_the_robber_on_the_desert(match):
    """The log's first event is the board after the first move, so the position
    everyone started from is reconstructed. Leaving the robber out of it would
    show a board with no robber on it, which is not a position Catan has."""
    replay = load(match["path"])
    opening = replay.frames[0]
    desert = next(t["coordinate"] for t in replay.meta["board"]["tiles"]
                  if t["tile"].get("type") == "DESERT")
    assert opening["robber"] == desert
    assert opening["buildings"] == {} and opening["roads"] == {}


def test_the_winner_reaches_ten_points_in_the_last_frame(match):
    """The frames are positions, not a re-simulation: the last one has to be the
    position the game actually ended in."""
    replay = load(match["path"])
    winner = replay.summary["winner"]
    assert replay.frames[-1]["players"][winner]["victory_points"] >= 10


def test_a_run_without_a_seating_order_is_refused_rather_than_guessed(tmp_path):
    """A log written before the order was recorded cannot be attributed at all.
    Refusing is the only honest option — an inferred mapping produces a complete,
    plausible, wrong replay."""
    path = tmp_path / "old.jsonl"
    start = {"seq": 0, "kind": "game_start", "game_id": "old", "seats": [], "board": {}}
    path.write_text(json.dumps(start) + "\n")
    with pytest.raises(ReplayError):
        load(path)


def test_a_file_still_being_written_never_yields_half_an_event(match, tmp_path):
    """The live case: the writer flushes and the reader reads, unsynchronised, so
    a read can land mid-line. Parsing that tail would either crash the viewer or —
    worse — drop the event silently."""
    whole = (tmp_path / "live.jsonl")
    source = open(match["path"], "rb").read()
    whole.write_bytes(b"")

    replay = Replay(whole)
    written = 0
    step = 997  # a size unrelated to any line length, so reads land mid-line
    while written < len(source):
        with whole.open("ab") as fh:
            fh.write(source[written:written + step])
        written += step
        replay.poll()  # must never raise, whatever it lands on

    assert len(replay.frames) == len(load(match["path"]).frames)
    assert replay.complete


def test_polling_returns_only_frames_that_are_new(match, tmp_path):
    """A live viewer asks repeatedly and must not be re-sent the whole match each
    time, nor shown a frame twice."""
    source = open(match["path"], "rb").read()
    half = source.rindex(b"\n", 0, len(source) // 2) + 1
    path = tmp_path / "partial.jsonl"

    path.write_bytes(source[:half])
    replay = Replay(path)
    first = replay.poll()
    assert first and not replay.complete

    with path.open("ab") as fh:
        fh.write(source[half:])
    second = replay.poll()

    assert second
    assert [f["seq"] for f in first][-1] < [f["seq"] for f in second][0]
    assert len(first) + len(second) == len(replay.frames)


def test_what_was_said_is_kept_apart_from_what_was_thought(match):
    """`say` is public and `reasoning` is private, and the whole point of the
    project is the gap between them. Merging the two in the viewer would destroy
    the only evidence that an agent bluffed."""
    replay = load(match["path"])
    pitches = [u for f in replay.frames for u in f["talk"] if u["kind"] == "pitch"]
    assert pitches, "a scripted match makes offers"
    assert all("text" in u and "reasoning" in u for u in pitches)
    assert any(u["text"] and u["reasoning"] and u["text"] != u["reasoning"]
               for u in pitches)


def test_a_pitch_carries_both_sides_of_the_offer(match):
    """A spectator has to see what was asked for, not only that something was."""
    replay = load(match["path"])
    pitch = next(u for f in replay.frames for u in f["talk"] if u["kind"] == "pitch")
    assert len(pitch["give"]) == 5 and len(pitch["want"]) == 5
    assert any(pitch["give"]) and any(pitch["want"])


def test_a_degraded_decision_surfaces_on_the_frame(tmp_path):
    """A model that keeps failing still produces a complete-looking match, because
    the reflex policy fills every gap. If the viewer cannot show that, it shows a
    persona that is not there."""
    path = tmp_path / "degraded.jsonl"
    lines = [
        {"seq": 0, "kind": "game_start", "game_id": "g", "seats": [],
         "order": ["RED", "BLUE"], "board": {}},
        {"seq": 1, "kind": "malformed", "color": "RED", "complaint": "empty reply",
         "attempt": 0},
        {"seq": 2, "kind": "fallback", "color": "RED",
         "action": {"type": "ROLL", "value": None, "described": "roll the dice"}},
        {"seq": 3, "kind": "state", "turn": 1, "current_color": "RED",
         "prompt": "PLAY_TURN", "robber": [0, 0, 0], "buildings": {}, "roads": {},
         "player_state": {}, "victory_points": {"RED": 0, "BLUE": 0}},
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))

    frame = load(path).frames[-1]
    assert {e["kind"] for e in frame["degraded"]} == {"malformed", "fallback"}


# ------------------------------------------- what the log must carry for any reader

def test_the_seating_order_is_logged(match):
    """Everything downstream reads hands out of the flat `P{i}_` namespace, and
    only `game_start` says which seat each index is. It is not derivable from the
    file otherwise."""
    start = read_run(match["path"])[0]
    assert start["kind"] == "game_start"
    assert sorted(start["order"]) == sorted(match["agents"].keys())


def test_the_engine_result_of_an_action_is_logged(match):
    """A decision event says what a seat chose; the roll, the steal and the drawn
    card are the engine's answer, and live only on the ActionRecord. Without them
    a replay shows the robber moving with no seven ever rolled."""
    states = [e for e in read_run(match["path"]) if e["kind"] == "state"]
    rolls = [e["last_action"] for e in states
             if e["last_action"] and e["last_action"]["type"] == "ROLL"]
    assert rolls, "a full match rolls dice"
    assert all(len(r["result"]) == 2 and all(1 <= d <= 6 for d in r["result"])
               for r in rolls)

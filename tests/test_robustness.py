"""A model will return an illegal move. The runner must absorb it.

Three failure shapes are covered: an index that is not a move, an offer of cards
the agent does not hold, and the call itself blowing up.
"""

from catanatron.models.player import Color
from catanatron.players.value import ValueFunctionPlayer

from arena.deciders import ScriptedDecider
from arena.events import read_run
from arena.personas import QUIET_BUILDER
from arena.runner import bot_seat, llm_seat, run_match
from tests.helpers import can_offer, d, index_of


def _run(script, tmp_path, seed=13):
    return run_match(
        seats=[
            (Color.RED, llm_seat(QUIET_BUILDER, ScriptedDecider(script))),
            (Color.BLUE, bot_seat(ValueFunctionPlayer)),
            (Color.ORANGE, bot_seat(ValueFunctionPlayer)),
        ],
        runs_dir=tmp_path,
        seed=seed,
    )


def test_out_of_range_choice_is_retried_then_accepted(tmp_path):
    """First answer is nonsense, second is the real move. One retry, no fallback."""
    state = {"bad": 0}

    def script(i, system, user):
        if state["bad"] < 1:
            state["bad"] += 1
            return d(999, reasoning="I will name a move that does not exist")
        return d(0)

    summary = _run(script, tmp_path)
    events = read_run(summary["path"])
    invalid = [e for e in events if e["kind"] == "invalid"]
    assert invalid, "the illegal index was never caught"
    assert "999 is not one of the moves" in invalid[0]["complaint"]
    assert invalid[0]["attempt"] == 0
    # the retry landed, so this decision never fell back
    assert any(e["kind"] == "decision" for e in events)


def test_persistent_nonsense_falls_back_to_reflex(tmp_path):
    summary = _run(lambda i, s, u: d(999), tmp_path)
    events = read_run(summary["path"])
    fallbacks = [e for e in events if e["kind"] == "fallback"]
    assert fallbacks, "the agent never fell back after two bad answers"
    assert summary["agents"]["RED"]["fallbacks"] > 0
    # and the game still finished rather than dying on the first bad answer
    assert events[-1]["kind"] == "game_over"
    assert summary["turns"] > 1


def test_offering_cards_you_do_not_hold_is_refused(tmp_path):
    """The engine would accept an impossible offer; we must not hand it one."""

    def script(i, system, user):
        if can_offer(user):
            return d(-1, give=["ore"] * 9, want=["wheat"], say="nine ore, take it")
        try:
            return d(index_of(user, "end your turn"))
        except AssertionError:
            return d(0)

    summary = _run(script, tmp_path)
    events = read_run(summary["path"])
    refused = [
        e for e in events
        if e["kind"] == "invalid" and "you do not hold what you offered" in e["complaint"]
    ]
    assert refused, "an unaffordable offer was allowed through"
    assert events[-1]["kind"] == "game_over"


def test_a_failing_call_does_not_stop_the_game(tmp_path):
    def script(i, system, user):
        raise RuntimeError("connection reset by peer")

    summary = _run(script, tmp_path)
    events = read_run(summary["path"])
    errors = [e for e in events if e["kind"] == "error"]
    assert errors and "connection reset" in errors[0]["detail"]
    assert any(e["kind"] == "fallback" for e in events)
    assert events[-1]["kind"] == "game_over"

"""Rebuilding a match in the engine: every position must be the one that was played.

The analyses built on this — what was legal, what the bot would have done,
whether a win was available — are only as good as the position they are run
in, and a position that drifted looks exactly as plausible as a right one.
"""

import json
import random

import pytest
from catanatron.models.enums import ActionType
from catanatron.models.player import Color
from catanatron.players.value import ValueFunctionPlayer

from arena.deciders import ScriptedDecider
from arena.events import read_run
from arena.personas import ROSTER
from arena.rebuild import RebuildError, plies, position
from arena.runner import bot_seat, llm_seat, run_match
from arena.scripted import heuristic_script


@pytest.fixture(scope="module")
def match(tmp_path_factory):
    """A scripted match with trades, robbers and a bot; GREY seated so the added
    colour is rebuilt too. Spends nothing."""
    def scripted(style, seed):
        return ScriptedDecider(heuristic_script(style, rng=random.Random(seed)),
                               label="dry-run")

    return run_match(
        seats=[
            (Color["GREY"], llm_seat(ROSTER["Plain"], scripted("trader", 1))),
            (Color.RED, llm_seat(ROSTER["Hard bargainer"], scripted("tough", 2))),
            (Color.BLUE, llm_seat(ROSTER["Cooperator"], scripted("trader", 3))),
            (Color.WHITE, bot_seat(ValueFunctionPlayer)),
        ],
        runs_dir=tmp_path_factory.mktemp("runs"), seed=21, max_offers_per_turn=3,
    )


def test_every_ply_rebuilds_to_the_snapshot_the_match_wrote(match):
    """`plies` raises on the first difference, so getting through the whole file
    means every position matched — and the rebuilt game ends where the match did."""
    last = None
    for last in plies(match["path"]):
        pass
    kinds = {e["last_action"]["type"] for e in read_run(match["path"])
             if e["kind"] == "state"}
    assert {"ROLL", "MOVE_ROBBER", "OFFER_TRADE", "CONFIRM_TRADE"} <= kinds, \
        "the fixture no longer exercises the moves that draw on the rng or trade"
    # exhausting the iterator applies the last action too
    assert last.game.winning_color().value == match["winner"]


def test_a_changed_dice_roll_is_caught_rather_than_replayed(match):
    """If a result were taken from the rng instead of the log, or the log were
    wrong, the rebuilt match would silently become a different game."""
    events = read_run(match["path"])
    roll = next(e for e in events[20:] if e["kind"] == "state"
                and e["last_action"]["type"] == "ROLL"
                and sum(e["last_action"]["result"]) != 7)
    a, b = roll["last_action"]["result"]
    roll["last_action"]["result"] = [a % 6 + 1, b] if a % 6 + 1 + b != 7 else [a, b % 6 + 1]
    with pytest.raises(RebuildError):
        for _ in plies(events):
            pass


def test_each_ply_carries_how_its_move_was_chosen(match):
    """A model's move carries its reasoning; the bot's and the reflex policy's do
    not. Mixing them up would attribute the bot's play, or a forced move, to a
    model's judgement."""
    seen = {}
    for ply in plies(match["path"]):
        seen.setdefault(ply.chosen_by, ply)
        if ply.chosen_by == "decision":
            assert ply.actor["color"] == ply.action.color.value
            assert ply.seat["kind"] == "llm"
        if ply.chosen_by == "bot":
            assert ply.seat["name"] == "ValueFunctionPlayer"
    assert {"decision", "reflex", "bot"} <= set(seen)


def test_a_position_can_be_played_forward_after_iteration_ends(match):
    """`plies` advances one live game; a position kept without copying would
    turn into the end of the match underneath whoever kept it."""
    ply = position(match["path"],
                   lambda p: p.action.action_type == ActionType.BUILD_CITY)
    assert ply is not None
    turn = ply.game.state.num_turns
    assert ply.action in ply.game.playable_actions
    ply.game.execute(ply.action)
    assert ply.game.state.num_turns == turn


def test_a_file_without_a_game_start_is_refused(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps({"kind": "state"}) + "\n")
    with pytest.raises(RebuildError):
        next(plies(path))


def test_a_rebuilt_prompt_is_exactly_the_prompt_the_agent_was_sent(tmp_path):
    """Re-asking a model about a recorded position only means something if it
    is asked the same question. The board, the hand, the table talk, the offer
    budget: any of them rebuilt differently would change the answer, and the
    difference would be blamed on the model. An odd offer budget is used so a
    rebuild that assumed the default would show."""
    sent = []

    def recording(style, seed):
        script = heuristic_script(style, rng=random.Random(seed))

        def wrapped(i, system, user):
            if "THAT DID NOT WORK" not in user:
                sent.append((system, user))
            return script(i, system, user)
        return ScriptedDecider(wrapped, label="dry-run")

    summary = run_match(
        seats=[
            (Color["GREY"], llm_seat(ROSTER["Plain"], recording("trader", 4))),
            (Color.RED, llm_seat(ROSTER["Hard bargainer"], recording("tough", 5))),
            (Color.BLUE, llm_seat(ROSTER["Cooperator"], recording("trader", 6))),
            (Color.WHITE, bot_seat(ValueFunctionPlayer)),
        ],
        runs_dir=tmp_path, seed=33, max_offers_per_turn=2,
    )
    rebuilt = [ply.prompt() for ply in plies(summary["path"])
               if ply.chosen_by in ("decision", "fallback")]
    assert len(rebuilt) == len(sent) > 100
    assert any("TABLE TALK (recent):\n  turn" in user for _, user in sent), \
        "the fixture never put talk in a prompt, so this proves nothing about it"
    for part in ("counters", "take up", "YOUR OWN NOTE", " got "):
        assert any(part in user for _, user in sent), \
            f"the fixture never showed {part!r}, so this proves nothing about it"
    for i, (got, want) in enumerate(zip(rebuilt, sent)):
        assert got == want, f"prompt {i} differs"

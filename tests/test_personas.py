"""Who sits where, and what the control is told."""

import json
import random
from pathlib import Path

from catanatron.models.player import Color
from catanatron.players.value import ValueFunctionPlayer

from arena.deciders import ScriptedDecider
from arena.personas import BOT_COLOR, COLORS, RULES, ROSTER, seat_colors
from arena.replay import load
from arena.runner import bot_seat, llm_seat, run_match
from arena.scripted import heuristic_script


def test_a_persona_keeps_its_colour_whatever_order_it_is_seated_in():
    """Agents address each other by colour and the viewer draws by colour. If the
    Cooperator were BLUE in one match and RED in the next, every comparison
    across matches — and any memory carried between them — would be attaching
    one player's history to another."""
    a, _ = seat_colors(["Hard bargainer", "Cooperator", "Quiet builder"])
    b, _ = seat_colors(["Quiet builder", "Cooperator", "Hard bargainer"])
    assert dict(zip(["Hard bargainer", "Cooperator", "Quiet builder"], a)) == \
        dict(zip(["Quiet builder", "Cooperator", "Hard bargainer"], b))
    for name, color in zip(["Hard bargainer", "Cooperator", "Quiet builder"], a):
        assert color == ROSTER[name].color


def test_the_bot_sits_on_white_at_every_table():
    """The bot is the ruler every persona is measured against, and the matches
    already played have it on WHITE. Moving it whenever the control sits down
    would make it a different player from one match to the next."""
    for table in (["Hard bargainer", "Cooperator", "Quiet builder"],
                  ["Plain", "Cooperator", "Hard bargainer"],
                  ["Plain", "Plain", "Quiet builder"]):
        personas, bots = seat_colors(table, bots=1)
        assert bots == [BOT_COLOR]
        assert BOT_COLOR not in personas


def test_a_second_copy_of_a_persona_gets_a_free_colour_not_a_shared_one():
    """Comparing two models on the same persona puts it at the table twice. Two
    seats on one colour would not even construct; the first copy keeps the
    persona's colour so it stays put across matches."""
    personas, bots = seat_colors(["Hard bargainer", "Hard bargainer", "Cooperator"], bots=1)
    assert personas[0] == "RED"
    assert len(set(personas + bots)) == 4
    assert set(personas + bots) <= set(COLORS)


def test_the_control_persona_is_told_the_rules_and_nothing_else():
    """Plain exists to show what the model does unprompted. Any style text that
    reaches it — even an empty YOUR STYLE header — makes the comparison with the
    styled personas measure two prompts against each other instead."""
    assert ROSTER["Plain"].system_prompt == RULES
    for persona in ROSTER.values():
        if persona.name != "Plain":
            assert persona.system_prompt.startswith(RULES)
            assert persona.instructions in persona.system_prompt


def test_every_persona_claims_a_different_colour():
    """Two personas on one colour would make seat_colors silently move whichever
    was listed second, and 'fixed colours' would hold only by luck of ordering."""
    claimed = [p.color for p in ROSTER.values()]
    assert len(claimed) == len(set(claimed))
    assert set(claimed) <= set(COLORS)


def test_a_grey_seat_plays_a_whole_match_and_replays_under_its_own_colour(tmp_path):
    """GREY is not one of Catanatron's colours; it is added to the engine's enum
    at import. If an upgrade of Python or the engine breaks that, this is where it
    should show — not as a crash forty turns into a paid match, and not as a
    replay that quietly shows GREY's hand under someone else's colour."""
    def scripted(seed):
        return ScriptedDecider(heuristic_script("trader", rng=random.Random(seed)),
                               label="dry-run")

    summary = run_match(
        seats=[
            (Color["GREY"], llm_seat(ROSTER["Plain"], scripted(1))),
            (Color.RED, llm_seat(ROSTER["Hard bargainer"], scripted(2))),
            (Color.WHITE, bot_seat(ValueFunctionPlayer)),
        ],
        runs_dir=tmp_path, seed=11, max_offers_per_turn=3,
    )
    assert summary["winner"] is not None
    assert summary["agents"]["GREY"]["llm_calls"] > 0
    final = load(summary["path"]).frames[-1]
    for color, agent in summary["agents"].items():
        assert final["players"][color]["victory_points"] == agent["victory_points"]


def test_a_saboteur_table_seats_three_plain_copies_and_black_without_moving_anyone(tmp_path):
    """The saboteur study compares {4 Plain} with {3 Plain + Saboteur}. BLACK is
    the second colour added to the engine; if it collided with a Plain copy, or
    a copy took BLACK, the replay would put one seat's hand under another's
    colour and the two tables would not differ in one seat only."""
    plain_only, _ = seat_colors(["Plain"] * 4)
    with_saboteur, _ = seat_colors(["Plain"] * 3 + ["Saboteur"])
    assert plain_only[:3] == with_saboteur[:3]
    assert with_saboteur[3] == "BLACK" and "BLACK" not in plain_only

    summary = run_match(
        seats=[(Color[c], llm_seat(ROSTER[p], ScriptedDecider(
                    heuristic_script("trader", rng=random.Random(i)), label="dry-run")))
               for i, (c, p) in enumerate(zip(with_saboteur, ["Plain"] * 3 + ["Saboteur"]))],
        runs_dir=tmp_path, seed=11, max_offers_per_turn=2,
    )
    assert summary["agents"]["BLACK"]["name"] == "Saboteur"
    assert summary["agents"]["BLACK"]["llm_calls"] > 0
    final = load(summary["path"]).frames[-1]
    for color, agent in summary["agents"].items():
        assert final["players"][color]["victory_points"] == agent["victory_points"]


def test_a_match_cut_at_the_turn_limit_says_so_and_names_no_winner(tmp_path):
    """A saboteur can stretch a match toward the engine's 1000 turns, each one
    paid for. The cap must end it, and the log must say it was cut rather than
    leave a winnerless match that reads like a crash or a draw."""
    summary = run_match(
        seats=[(Color[c], llm_seat(ROSTER["Plain"], ScriptedDecider(
                    heuristic_script("trader", rng=random.Random(i)), label="dry-run")))
               for i, c in enumerate(["GREY", "RED", "BLUE"])],
        runs_dir=tmp_path, seed=11, max_turns=12,
    )
    assert summary["truncated"] and summary["winner"] is None
    assert summary["turns"] == 12
    start = json.loads(Path(summary["path"]).open().readline())
    assert start["max_turns"] == 12


def test_the_effort_each_seat_played_at_is_in_the_log(tmp_path):
    """Luna finds half again as many wins at 'high' as at its default, so a
    match at 'high' belongs to a different group. Without the effort on the
    seat, the two would be pooled by anyone reading the runs later."""
    class AtHigh(ScriptedDecider):
        effort = "high"

    summary = run_match(
        seats=[(Color.RED, llm_seat(ROSTER["Plain"], AtHigh(
                    heuristic_script("trader", rng=random.Random(1)), label="dry-run"))),
               (Color.WHITE, bot_seat(ValueFunctionPlayer))],
        runs_dir=tmp_path, seed=11, max_turns=5,
    )
    seats = json.loads(Path(summary["path"]).open().readline())["seats"]
    assert {s["color"]: s["effort"] for s in seats} == {"RED": "high", "WHITE": None}


def test_a_match_asked_for_flex_gets_it_with_the_room_flex_needs(monkeypatch):
    """`--flex` halves a match's bill. If the flag were dropped on its way to the
    decider, a batch would pay full price while its operator believed otherwise;
    if the 'high' budget's 300 s timeout replaced flex's 15 minutes, slow flex
    answers would be cut off and turned into reflex moves."""
    import arena.play as play
    built, ran = {}, {}

    def fake_make_decider(model, **kwargs):
        built.update(kwargs, model=model)
        return ScriptedDecider(heuristic_script("trader", rng=random.Random(1)), label=model)

    monkeypatch.setattr(play, "make_decider", fake_make_decider)
    monkeypatch.setattr(play, "run_match", lambda **kw: ran.update(kw) or {
        "winner": None, "turns": 0, "seconds": 0, "offers": 0, "agents": {}, "path": "-"})
    monkeypatch.setattr(play, "cost_report", lambda summary: "")
    monkeypatch.setenv("OPENAI_API_KEY", "unused")
    play.main(["--model", "gpt-5.6-luna", "--effort", "high", "--flex", "--quiet",
               "--seat", "Plain", "--seat", "Saboteur"])
    assert built["flex"] is True and built["max_tokens"] == 16384
    assert "timeout" not in built


def test_flex_on_a_model_without_it_stops_before_the_match_starts(monkeypatch):
    """DeepSeek has no flex tier. A match that silently ran at full price, or
    crashed after the board was dealt, would both be worse than refusing."""
    import pytest
    import arena.play as play
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unused")
    with pytest.raises(SystemExit):
        play.main(["--model", "deepseek-flash", "--flex", "--seat", "Plain", "--seat", "Plain"])

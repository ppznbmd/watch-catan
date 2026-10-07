"""Who sits where, and what the control is told."""

import random

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

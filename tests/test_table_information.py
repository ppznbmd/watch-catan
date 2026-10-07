"""What an agent sees that anybody at a physical table sees.

Roads, ports, the dice and the choice of discard were all missing until
2026-09-18 (DESIGN.md, *What an agent knows, against a player at a real table*).
Losing any of them again would be silent: matches still run, agents still
answer, and they go back to choosing roads without a map.
"""

import json
import random
import re

import pytest
from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import ActionPrompt
from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS, Port
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.state_functions import get_player_buildings

from arena.colors import FULL_TABLE
from arena.deciders import ScriptedDecider
from arena.personas import ROSTER
from arena.prompt import build_prompt
from arena.runner import bot_seat, llm_seat, run_match
from arena.scripted import heuristic_script
from arena.table_talk import TableTalk


def midgame(seed=5, ticks=120):
    game = Game(players=[RandomPlayer(c) for c in list(FULL_TABLE)], seed=seed)
    for _ in range(ticks):
        game.play_tick()
    return game


def prompt_for(game, color):
    game.state.current_player_index = game.state.colors.index(color)
    game.state.current_prompt = ActionPrompt.PLAY_TURN
    return build_prompt(game, color, generate_playable_actions(game.state), TableTalk(),
                        may_offer=False)


def section(text, header):
    """The lines of one prompt section, up to the blank line that ends it."""
    body = text.split(f"\n{header}", 1)[1].split("\n", 1)[1]
    return body.split("\n\n", 1)[0].splitlines()


@pytest.mark.parametrize("seat", range(4))
def test_every_road_on_the_board_is_shown_to_every_seat(seat):
    """A player at a table sees every road, a rival's included. Without them an
    agent cannot tell where a rival is heading or which spot is about to be cut
    off, and its only view of its own network is the list of legal road moves."""
    game = midgame()
    me = game.state.colors[seat]
    lines = section(prompt_for(game, me), "ROADS")
    assert sum(len(get_player_buildings(game.state, c, "ROAD"))
               for c in game.state.colors) > 8, "fixture has too few roads to prove much"
    for color in game.state.colors:
        line = next(l for l in lines if l.strip().startswith(color.value))
        shown = {tuple(map(int, m)) for m in re.findall(r"\((\d+), (\d+)\)", line)}
        built = {tuple(sorted(e)) for e in get_player_buildings(game.state, color, "ROAD")}
        assert shown == built, f"{color.value}'s roads shown to {me.value} as {line}"


def test_every_port_is_shown_with_its_nodes_and_whoever_sits_on_it():
    """Before this an agent saw only the ports it already owned, so it could
    never plan a road toward one."""
    game = midgame()
    me = game.state.colors[0]
    lines = section(prompt_for(game, me), "PORTS")
    ports = [t for t in game.state.board.map.tiles.values() if isinstance(t, Port)]
    assert len(lines) == len(ports) == 9
    for tile in ports:
        a, b = PORT_DIRECTION_TO_NODEREFS[tile.direction]
        lo, hi = sorted((tile.nodes[a], tile.nodes[b]))
        kind = f"2:1 {tile.resource.lower()}" if tile.resource else "3:1"
        line = next(l for l in lines if f"{kind} at nodes {lo} and {hi}" in l)
        for node in (lo, hi):
            building = game.state.board.buildings.get(node)
            if building:
                assert f"{building[0].value} {building[1].lower()} at {node}" in line


# ------------------------------------------------------------- in a played match


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """A dry match: its event log, and every first-attempt prompt a seat was sent."""
    sent = []

    def recording(style, rng_seed):
        script = heuristic_script(style, rng=random.Random(rng_seed))

        def wrapped(i, system, user):
            if "THAT DID NOT WORK" not in user:
                sent.append(user)
            return script(i, system, user)
        return ScriptedDecider(wrapped, label="dry-run")

    summary = run_match(
        seats=[
            (Color.RED, llm_seat(ROSTER["Hard bargainer"], recording("tough", 1))),
            (Color.BLUE, llm_seat(ROSTER["Cooperator"], recording("trader", 2))),
            (Color.ORANGE, llm_seat(ROSTER["Quiet builder"], recording("quiet", 3))),
            (Color.WHITE, bot_seat(ValueFunctionPlayer)),
        ],
        runs_dir=tmp_path_factory.mktemp("runs"), seed=11, max_offers_per_turn=3,
    )
    events = [json.loads(line) for line in open(summary["path"])]
    return events, sent


def seat_of(user):
    return re.match(r"TURN \d+ — you are (\w+)\.", user).group(1)


HISTORY = "SINCE YOUR LAST DECISION"


def test_each_seat_is_shown_every_roll_exactly_once(recorded):
    """The model is not called on anybody else's roll, so each prompt carries the
    events since that seat's previous one. Read in order, a seat's prompts must
    replay the game's dice with nothing skipped and nothing repeated — a marker
    moved in the wrong place would do one or the other, and the prompts would
    still look entirely plausible."""
    events, sent = recorded
    rolls = [(e["last_action"]["color"], sum(e["last_action"]["result"]))
             for e in events if e["kind"] == "state" and e["last_action"]["type"] == "ROLL"]
    assert len(rolls) > 40
    for color in ("RED", "BLUE", "ORANGE"):
        shown = []
        for user in (u for u in sent if seat_of(u) == color):
            lines = section(user, HISTORY)
            assert not any("not shown" in l for l in lines), "the cap cut a real gap"
            shown += [(color if m.group(1) == "you" else m.group(1), int(m.group(2)))
                      for l in lines if (m := re.match(r"\s+(\w+) rolled (\d+)", l))]
        assert shown == rolls[:len(shown)], f"{color} was shown the dice out of order"
        # The last prompt a seat saw came after all but the last round or so.
        assert len(rolls) - len(shown) <= 4, f"{color} missed rolls: {len(shown)}/{len(rolls)}"


def test_what_a_roll_paid_out_is_what_the_hands_gained(recorded):
    """Production is not in the engine's record of a roll, so it is taken as the
    change in every hand around it. Checked against the omniscient snapshots
    the match wrote: a wrong attribution would teach an agent a false count of
    its rivals' cards."""
    events, sent = recorded
    order = events[0]["order"]
    states = [e for e in events if e["kind"] == "state"]
    paid = {}
    for before, after in zip(states, states[1:]):
        la = after["last_action"]
        if la["type"] != "ROLL" or sum(la["result"]) == 7:
            continue
        got = {}
        for i, c in enumerate(order):
            d = [after["player_state"][f"P{i}_{r}_IN_HAND"] - before["player_state"][f"P{i}_{r}_IN_HAND"]
                 for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")]
            if any(d):
                got[c] = d
        paid[(after["turn"], la["color"])] = got
    checked = 0
    for user in sent:
        me = seat_of(user)
        turn = int(re.match(r"TURN (\d+)", user).group(1))
        for l in section(user, HISTORY):
            m = re.match(r"\s+(\w+) rolled (\d+): (.*)", l)
            if not m:
                continue
            roller = me if m.group(1) == "you" else m.group(1)
            want = {}
            for part in ([] if m.group(3) == "nobody produced" else m.group(3).split("; ")):
                who, cards = part.split(" got ")
                deck = [0] * 5
                for c in cards.split(", "):
                    n, r = c.split()
                    deck[("wood", "brick", "sheep", "wheat", "ore").index(r)] = int(n)
                want[me if who == "you" else who] = deck
            candidates = [v for (t, c), v in paid.items() if c == roller and t <= turn]
            assert want in candidates, f"{l} matches no roll by {roller}"
            checked += 1
    assert checked > 100


def test_a_stolen_card_is_named_only_to_thief_and_victim(recorded):
    """At a table the robber's victim hands over a card face down. Naming it to a
    third player would leak a hand."""
    _, sent = recorded
    named = unnamed = 0
    for user in sent:
        me = seat_of(user)
        for l in section(user, HISTORY):
            m = re.match(r"\s+(\w+) moved the robber to .* stole (1 \w+|a card) from (\w+)", l)
            if not m:
                continue
            involved = "you" in (m.group(1), m.group(3))
            if m.group(2) == "a card":
                unnamed += 1
            else:
                assert involved, f"{me} was told what was stolen: {l}"
                named += 1
    assert named and unnamed


def test_a_seat_is_told_its_own_last_note_and_nobody_elses(recorded):
    """A player remembers what it was trying to do; a model call does not. The
    note is private: another seat's reasoning in a prompt would be a leak of the
    worst kind, the plan itself."""
    events, sent = recorded
    own = {}
    for e in events:
        if e["kind"] == "decision":
            own.setdefault(e["color"], set()).add(e["reasoning"])
    notes = 0
    for user in sent:
        m = re.search(r"YOUR OWN NOTE [^\n]*\n  ([^\n]*)", user)
        if m:
            me = seat_of(user)
            assert m.group(1) in own[me]
            notes += 1
    assert notes > 50


def test_a_discard_with_a_real_choice_goes_to_the_model(recorded):
    """Which cards to lose after a 7 used to be picked at random for the agent.
    It is a strategic choice, so it must reach the model, told how many remain."""
    events, sent = recorded
    discards = [u for u in sent if "\nDISCARD: " in u]
    assert discards, "the fixture never made a seat discard with a choice"
    for user in discards:
        assert re.search(r"You must discard \d+ more card\(s\)", user)
        moves = [l for l in user.splitlines() if re.match(r"\s*\[\d+\] discard 1 ", l)]
        assert len(moves) >= 2
    chosen = [e for e in events if e["kind"] == "decision"
              and e["action"]["type"] == "DISCARD_RESOURCE"]
    assert len(chosen) == len(discards)

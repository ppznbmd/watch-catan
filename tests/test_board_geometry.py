"""Whether an agent can tell one node from another.

The board section names tiles by cube coordinate and the moves name nodes by
integer id, and for a whole session nothing joined the two. An agent picking its
opening settlement was choosing between 54 interchangeable integers, which is
the single largest thing standing between the table and the bot: a value
function that plays perfectly except for a blind opening still loses 82 of 100
matches to the same bot, against 25 of 100 for a fair seat.

These tests fail loudly rather than let the join rot, because losing it is
invisible — the matches still run, the agents still answer, and the openings
quietly go back to being random.
"""

import re

import pytest
from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import ActionPrompt, ActionType
from catanatron.models.player import Color, RandomPlayer

from arena.colors import FULL_TABLE
from arena.prompt import (
    PIPS,
    annotate_action,
    build_prompt,
    describe_node,
    node_neighbours,
    ports_by_node,
    reached_nodes,
)
from arena.scripted import find, legal_moves
from arena.table_talk import TableTalk


def opening(seed=7):
    """A fresh game, sitting on the first initial-settlement decision."""
    game = Game([RandomPlayer(c) for c in list(FULL_TABLE)], seed=seed)
    assert game.state.current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT
    return game


def opening_prompt(game):
    me = game.state.current_color()
    actions = generate_playable_actions(game.state)
    return build_prompt(game, me, actions, TableTalk(), may_offer=False)


def played(seed, ticks):
    """A game a few turns in, where roads and the distance rule are in play."""
    game = Game([RandomPlayer(c) for c in list(FULL_TABLE)], seed=seed)
    for _ in range(ticks):
        if game.winning_color():
            break
        game.play_tick()
    return game


# ------------------------------------------------------------------ settlements


def test_a_settlement_move_says_what_the_node_produces():
    """Without this the opening is a coin flip: every one of the 54 options
    reads the same, and the agent's own reasoning degrades to picking the first
    one on the list."""
    text = opening_prompt(opening())
    lines = [d for _, d in legal_moves(text) if d.startswith("build a settlement")]
    assert len(lines) == 54, "the opening should offer every land node"
    for line in lines:
        assert re.search(r"\(\d+ pips: ", line), f"no production on: {line}"


def test_two_nodes_that_differ_are_described_differently():
    """The join is worth nothing if it renders every node the same. A real board
    has 13-pip corners and 0-pip ones, and the agent has to see the spread."""
    game = opening()
    ports = ports_by_node(game.state)
    described = {
        node: describe_node(game.state, node, ports)
        for node in game.state.board.map.land_nodes
    }
    pips = {int(re.match(r"(\d+) pips", d).group(1)) for d in described.values()}
    assert len(described) == 54
    assert len(set(described.values())) > 40, "nodes are being flattened together"
    assert max(pips) - min(pips) >= 8, f"implausibly flat board: {sorted(pips)}"


@pytest.mark.parametrize("seed", [1, 4, 7])
def test_the_pips_of_a_node_match_the_engines_own_production_table(seed):
    """Pips are hand-summed from the tile numbers while the engine keeps
    `node_production` in probabilities. If the two ever disagree the agent is
    being told a number the board does not support, which is worse than telling
    it nothing."""
    game = Game([RandomPlayer(c) for c in list(FULL_TABLE)], seed=seed)
    ports = ports_by_node(game.state)
    for node in game.state.board.map.land_nodes:
        mine = int(re.match(r"(\d+) pips", describe_node(game.state, node, ports)).group(1))
        engine = round(sum(game.state.board.map.node_production[node].values()) * 36)
        assert mine == engine, f"node {node}: rendered {mine} pips, engine says {engine}"


def test_the_desert_contributes_nothing():
    """A corner on the desert produces on no roll at all. Counting it would make
    the worst spot on the board look ordinary."""
    game = opening()
    desert = next(t for t in game.state.board.map.land_tiles.values() if t.resource is None)
    for node in desert.nodes.values():
        rendered = describe_node(game.state, node)
        assert "desert" in rendered
        neighbours_pips = sum(
            PIPS.get(t.number, 0)
            for t in game.state.board.map.adjacent_tiles[node]
            if t.resource is not None
        )
        assert rendered.startswith(f"{neighbours_pips} pips:")


def test_a_port_node_is_marked_as_a_port():
    """A 2:1 port is a large part of what makes a low-production corner worth
    taking, and it is invisible in the tile list."""
    game = opening()
    ports = ports_by_node(game.state)
    assert ports, "the map has ports; the lookup found none"
    for node, label in ports.items():
        assert f"{label} port" in describe_node(game.state, node, ports)


def test_the_robber_is_marked_on_the_tile_it_sits_on():
    """The board section marks the robber on a coordinate, which an agent can no
    more match to a node than anything else. An unmarked robber makes a dead
    tile read as a live one.

    Both placements are checked on purpose. The robber starts on the desert and
    stays there until somebody rolls a seven, so a test that only looks at a
    fresh game exercises the desert branch and lets the resource branch — the
    one that actually costs an agent production — rot unnoticed. It did: this
    test passed with the resource marking deleted.
    """
    game = opening()
    desert = next(t for t in game.state.board.map.land_tiles.values() if t.resource is None)
    assert game.state.board.robber_coordinate in game.state.board.map.land_tiles
    assert "desert [robber]" in describe_node(game.state, next(iter(desert.nodes.values())))

    coord, tile = next(
        (c, t) for c, t in game.state.board.map.land_tiles.items() if t.resource is not None
    )
    game.state.board.robber_coordinate = coord  # what rolling a seven does
    for node in tile.nodes.values():
        rendered = describe_node(game.state, node)
        assert f"{tile.resource.lower()} {tile.number} [robber]" in rendered, rendered
    # and nowhere else
    away = next(
        n for n in game.state.board.map.land_nodes
        if tile not in game.state.board.map.adjacent_tiles[n]
    )
    assert "[robber]" not in describe_node(game.state, away)


# ----------------------------------------------------------------------- roads


def turns_offering(seed, action_type, want, max_ticks=2000):
    """Walk a game and stop at each state where this move is on the table.

    Sampling every N ticks finds almost nothing: a road is legal only post-roll,
    on your own turn, holding wood and brick. The first draft of these tests
    sampled and passed vacuously on zero roads.
    """
    game = Game([RandomPlayer(c) for c in list(FULL_TABLE)], seed=seed)
    seen = 0
    for _ in range(max_ticks):
        if game.winning_color():
            break
        actions = generate_playable_actions(game.state)
        matching = [a for a in actions if a.action_type == action_type]
        if matching:
            yield game, matching
            seen += 1
            if seen >= want:
                return
        game.play_tick()


def road_lines(game, roads):
    me = game.state.current_color()
    ports, neighbours = ports_by_node(game.state), node_neighbours(game.state)
    return [(a, annotate_action(game.state, a, me, ports, neighbours)) for a in roads]


@pytest.mark.parametrize("seed", [2, 5, 9])
def test_a_road_move_says_which_node_it_reaches(seed):
    """A road is only ever worth building for where it lets you settle. Rendered
    as a bare pair of integers it is a coin flip with extra steps."""
    checked = 0
    for game, roads in turns_offering(seed, ActionType.BUILD_ROAD, want=12):
        reached = reached_nodes(game.state, game.state.current_color())
        for action, line in road_lines(game, roads):
            assert "reaching node " in line, f"road with no destination: {line}"
            far = [n for n in action.value if n not in reached] or list(action.value)
            for node in far:
                assert f"node {node} (" in line, f"node {node} missing from: {line}"
            checked += 1
    assert checked, f"seed {seed} never offered a road to check"


@pytest.mark.parametrize("seed", [2, 5, 9])
def test_a_road_never_promises_a_node_the_distance_rule_forbids(seed):
    """Buildings are never removed, so a node next to one is blocked for the
    rest of the game. Sending an agent down a road toward a corner it can never
    settle is worse than saying nothing about the road at all."""
    checked = 0
    for game, roads in turns_offering(seed, ActionType.BUILD_ROAD, want=12):
        state = game.state
        neighbours = node_neighbours(state)
        for action, line in road_lines(game, roads):
            for node in action.value:
                if f"node {node} (" not in line:
                    continue
                blocked = node in state.board.buildings or any(
                    other in state.board.buildings for other in neighbours[node]
                )
                warned = re.search(
                    rf"node {node} \([^)]*\), where nobody may ever settle", line
                )
                assert bool(warned) == blocked, (
                    f"node {node} blocked={blocked} but rendered as: {line}"
                )
                checked += 1
    assert checked, f"seed {seed} never offered a road to check"


# ------------------------------------------------------------- the dry run surface


def test_the_scripted_agent_still_finds_its_moves():
    """`arena/scripted.py` locates its moves by the rendered description, so the
    dry run is the proof that the render stays legible. Annotations are appended
    for exactly this reason: a prefix match keeps working, a rewrite would break
    every dry run and every test that picks a move by name."""
    text = opening_prompt(opening())
    assert find(text, "build a settlement") == 0

    checked = 0
    for game, _ in turns_offering(4, ActionType.BUILD_ROAD, want=3):
        me = game.state.current_color()
        actions = generate_playable_actions(game.state)
        mid = build_prompt(game, me, actions, TableTalk(), may_offer=False)
        assert find(mid, "build a road") is not None
        checked += 1
    assert checked, "no road turn found to check the dry-run surface against"


def test_the_annotations_carry_nothing_an_agent_could_not_see_on_the_board():
    """Everything added here is public: tiles, numbers, ports, the robber, and
    who has built where. If a hand or a dev card ever leaks in through this
    path, every negotiation in the project becomes theatre."""
    game = played(6, 80)
    me = game.state.current_color()
    actions = generate_playable_actions(game.state)
    ports, neighbours = ports_by_node(game.state), node_neighbours(game.state)
    rendered = " ".join(
        annotate_action(game.state, a, me, ports, neighbours) for a in actions
    )
    banned = ("hand", "holds", "dev card", "knight in", "victory point card")
    for word in banned:
        assert word not in rendered.lower(), f"{word!r} leaked into a move description"

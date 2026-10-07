"""What an agent may not know.

The whole project rests on this: if an agent can read an opponent's hand, every
negotiation is theatre. These tests fail loudly rather than let that regress
quietly, because a leak is invisible in the output — the matches would still look
fine and mean nothing.
"""

import re

import pytest
from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES, ActionPrompt
from catanatron.models.player import Color, RandomPlayer
from catanatron.state_functions import player_key

from arena.colors import FULL_TABLE
from arena.prompt import build_prompt
from arena.table_talk import TableTalk

# One per seat, all distinct — two seats sharing a fingerprint would make the
# test accuse the prompt of leaking what is actually the reader's own hand.
FINGERPRINTS = [("ORE", 7), ("BRICK", 9), ("SHEEP", 8), ("WOOD", 6)]


def rigged_game():
    """A mid-game state where every seat holds an unmistakable hand."""
    game = Game(players=[RandomPlayer(c) for c in list(FULL_TABLE)], seed=5)
    for _ in range(40):
        game.play_tick()
    assert len(FINGERPRINTS) >= len(game.state.colors)
    for color, (resource, n) in zip(game.state.colors, FINGERPRINTS):
        key = player_key(game.state, color)
        # Set the hand, never add to it. Those 40 ticks leave each seat holding
        # whatever the engine's action ordering produced, and that ordering
        # varies per process (DESIGN.md, *Reproducibility*). Adding 7 ore to a
        # seat that already held 2 makes the prompt say "9 ore" and the
        # fingerprint stops matching — this failed on roughly one hash seed in
        # six, which is the worst way for a test of this invariant to behave.
        for other in RESOURCES:
            game.state.player_state[f"{key}_{other}_IN_HAND"] = (
                n if other == resource else 0)
        for card in DEVELOPMENT_CARDS:
            game.state.player_state[f"{key}_{card}_IN_HAND"] = 0
        game.state.player_state[f"{key}_KNIGHT_IN_HAND"] = 2
        # a victory point card: raises actual VPs without raising visible ones
        game.state.player_state[f"{key}_VICTORY_POINT_IN_HAND"] = 1
        game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] = (
            game.state.player_state[f"{key}_VICTORY_POINTS"] + 1)
    return game


def prompt_for(game, color):
    game.state.current_player_index = game.state.colors.index(color)
    game.state.current_prompt = ActionPrompt.PLAY_TURN
    actions = generate_playable_actions(game.state)
    return build_prompt(game, color, actions, TableTalk(), may_offer=True)


def player_line(text, color):
    """The PLAYERS line for `color`. Other sections (ROADS) also open lines with
    a colour, so the search starts at the section header."""
    lines = text.split("\nPLAYERS:\n", 1)[1].splitlines()
    return next(l for l in lines if l.strip().startswith(color.value))


@pytest.mark.parametrize("seat", range(4))
def test_an_agent_cannot_read_an_opponents_hand(seat):
    game = rigged_game()
    me = game.state.colors[seat]
    text = prompt_for(game, me).lower()

    # Whole numbers only: the bank line says "16 wood", which is not "6 wood".
    def shows(n, resource):
        return re.search(rf"\b{n} {resource.lower()}\b", text) is not None

    mine = FINGERPRINTS[seat]
    assert shows(mine[1], mine[0]), "an agent must see its own hand exactly"

    for other_seat, color in enumerate(game.state.colors):
        if color == me:
            continue
        resource, n = FINGERPRINTS[other_seat]
        assert not shows(n, resource), (
            f"{me.value} can read {color.value}'s hand composition"
        )


@pytest.mark.parametrize("seat", range(4))
def test_an_agent_cannot_read_an_opponents_development_cards(seat):
    game = rigged_game()
    me = game.state.colors[seat]
    text = prompt_for(game, me).lower()

    # Every seat holds the same dev cards in this fixture, so counting how many
    # lines *name card types* is the test: exactly one, the reader's own.
    named = [l for l in text.splitlines() if "dev cards:" in l and "knight" in l]
    assert len(named) == 1, f"a seat other than {me.value} had its cards named: {named}"
    assert "2 knight" in named[0] and "1 victory_point" in named[0]

    # everyone else gets a bare count
    counts = [l for l in text.splitlines() if "dev cards:" in l and "knight" not in l]
    assert len(counts) == len(game.state.colors) - 1
    assert all("dev cards: 3" in l for l in counts), counts


@pytest.mark.parametrize("seat", range(4))
def test_a_hidden_victory_point_stays_hidden(seat):
    """A victory point card raises actual VPs but not visible ones. An agent sees
    its own actual total and everyone else's visible total — otherwise it knows
    exactly who is one card from winning."""
    from catanatron.state_functions import (
        get_actual_victory_points,
        get_visible_victory_points,
    )

    game = rigged_game()
    me = game.state.colors[seat]
    text = prompt_for(game, me)

    for color in game.state.colors:
        actual = get_actual_victory_points(game.state, color)
        visible = get_visible_victory_points(game.state, color)
        assert actual > visible, "fixture did not create a hidden victory point"
        expected = actual if color == me else visible
        line = player_line(text, color)
        assert f"{expected} VP" in line, (
            f"{color.value} shown as the wrong total to {me.value}: {line.strip()}"
        )


@pytest.mark.parametrize("seat", range(4))
def test_an_agent_is_told_its_total_already_counts_its_hidden_point(seat):
    """Shown a bare "8 VP" beside "1 victory_point", a model in a real match read
    it as eight visible plus one hidden and reasoned about a ninth point it did
    not have. The split has to be stated, and must add up to the real total."""
    game = rigged_game()
    me = game.state.colors[seat]
    line = player_line(prompt_for(game, me), me)
    from catanatron.state_functions import (
        get_actual_victory_points,
        get_visible_victory_points,
    )
    actual = get_actual_victory_points(game.state, me)
    visible = get_visible_victory_points(game.state, me)
    assert (f"{actual} VP ({visible} that everyone can see + {actual - visible} "
            f"from victory point cards only you can see)") in line


@pytest.mark.parametrize("seat", range(4))
def test_road_length_and_played_knights_are_public_and_knights_in_hand_are_not(seat):
    """Longest road and largest army are decided by numbers everyone at a real
    table can see. Without the road length, a model twice skipped the road that
    would have won it the game. Knights still in hand are secret, so the count
    shown must be the played ones — the fixture gives every seat two unplayed."""
    game = rigged_game()
    for i, color in enumerate(game.state.colors):
        key = player_key(game.state, color)
        game.state.player_state[f"{key}_LONGEST_ROAD_LENGTH"] = 3 + i
        game.state.player_state[f"{key}_PLAYED_KNIGHT"] = 5 + i
    me = game.state.colors[seat]
    text = prompt_for(game, me)
    for i, color in enumerate(game.state.colors):
        line = player_line(text, color)
        assert f"longest continuous road {3 + i}, {5 + i} knights played" in line


def test_table_talk_carries_speech_and_never_reasoning():
    """`reasoning` is written for the spectator. If it reached the transcript,
    every agent would read every opponent's private deliberation."""
    talk = TableTalk()
    talk.record_pitch(3, "RED", (1, 0, 0, 0, 0), (0, 0, 0, 0, 1), "wood for ore")
    talk.record(3, "BLUE", "reply", "no thanks")
    assert set(vars(talk.utterances[0])) == {
        "turn", "color", "kind", "text", "give", "want", "to"
    }, "Utterance grew a field — check it is not private to one agent"
    transcript = talk.transcript(3)
    assert "wood for ore" in transcript and "no thanks" in transcript

"""Turning engine state into something an agent can read, and reading its answer back.

Two rules hold this together:

- Hidden information stays hidden. An agent sees its own hand exactly and every
  opponent's hand *size* only. Handing the model the full state would make every
  negotiation a formality.
- The model never names a raw action. It picks an index out of a list we built
  from `playable_actions`, or authors a trade offer. An index cannot be illegal.
"""

from typing import List, Optional, Tuple

from catanatron.models.enums import RESOURCES, ActionType, ActionPrompt
from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS, LandTile, Port
from catanatron.state_functions import (
    get_actual_victory_points,
    get_dev_cards_in_hand,
    get_largest_army,
    get_longest_road_color,
    get_longest_road_length,
    get_played_dev_cards,
    get_player_buildings,
    get_player_freqdeck,
    get_visible_victory_points,
    player_num_dev_cards,
    player_num_resource_cards,
)

from arena.table_talk import freqdeck_to_text

AUTHOR_OFFER = -1  # the choice index meaning "I am writing my own trade offer"

#: What the agents are shown, as a whole. Bump it with any change to the prompt
#: that could change a decision: it is logged in every run's game_start, and a
#: comparison across versions compares two different games.
#:   1  the first four real matches
#:   2  + own VP split out, road length and knights played (the 2026-09-18 batch)
#:   3  + roads, ports, public history since the last decision, bank and deck,
#:      the agent's own last note, counter-offers, discards, 2 offers a turn,
#:      and a bot that answers offers on their merits
#:   4  `reasoning` asked for as a `note` to itself: a summary requested for a
#:      reader told the model it had one
PROMPT_VERSION = 4


# ---------------------------------------------------------------- board & state


def render_board(state) -> str:
    lines = []
    robber = state.board.robber_coordinate
    for coord, tile in sorted(state.board.map.tiles.items()):
        if not isinstance(tile, LandTile):
            continue
        if tile.resource is None:
            desc = "desert"
        else:
            desc = f"{tile.resource.lower()} {tile.number}"
        mark = "  <- ROBBER" if coord == robber else ""
        owners = []
        for node_id in tile.nodes.values():
            b = state.board.buildings.get(node_id)
            if b:
                owners.append(f"{b[0].value}:{b[1].lower()}@{node_id}")
        owned = f"  [{', '.join(sorted(set(owners)))}]" if owners else ""
        lines.append(f"  {coord} {desc}{owned}{mark}")
    return "\n".join(lines)


def render_roads(state, me) -> str:
    """Every road on the board, by owner.

    Public at any table, and until 2026-09-18 missing from the prompt: an agent
    saw road counts and lengths but not where anyone's network ran, its own
    included, so it could not tell where a rival was heading or which spot was
    about to be cut off. Edges use the same `(a, b)` form as the road moves.
    """
    lines = []
    for color in state.colors:
        edges = sorted(tuple(sorted(e)) for e in get_player_buildings(state, color, "ROAD"))
        # Not "(you)": that marks the one PLAYERS line holding the agent's hand,
        # and `arena/scripted.py` finds the hand by it.
        who = f"{color.value}, yours" if color == me else color.value
        lines.append(f"  {who}: {', '.join(map(str, edges)) or 'none'}")
    return "\n".join(lines)


def render_ports(state) -> str:
    """Every port, the two nodes that use it, and who already sits there.

    Before this an agent saw only the ports it owned, and a new one only when
    settling on it happened to be a legal move that turn — so it could never
    plan a road toward one.
    """
    lines = []
    for tile in state.board.map.tiles.values():
        if not isinstance(tile, Port):
            continue
        a, b = PORT_DIRECTION_TO_NODEREFS[tile.direction]
        nodes = sorted((tile.nodes[a], tile.nodes[b]))
        kind = f"2:1 {tile.resource.lower()}" if tile.resource else "3:1"
        held = [f"{state.board.buildings[n][0].value} {state.board.buildings[n][1].lower()} at {n}"
                for n in nodes if n in state.board.buildings]
        tail = f" ({', '.join(held)})" if held else ""
        lines.append((nodes, f"  {kind} at nodes {nodes[0]} and {nodes[1]}{tail}"))
    return "\n".join(line for _, line in sorted(lines))


#: A round is typically 10-25 public events. The cap only bounds the prompt if a
#: seat goes unasked for a long stretch.
MAX_EVENTS_SHOWN = 40


def _who(color, me) -> str:
    return "you" if color == me else color.value


def render_history(state, me, since: int, talk) -> str:
    """Everything a player at the table saw happen since this seat was last asked.

    The model is not called on anybody else's roll, build or trade — there is
    nothing for it to decide — so without this it saw its hand change with no
    idea why, and could not count what its rivals had collected. `since` is an
    index into `state.action_records`: where the record stood at this seat's
    previous prompt.

    Offers and replies are left to TABLE TALK. A stolen card is named only to
    the two players it passed between, as at a real table.
    """
    lines = []
    records = state.action_records
    i = since
    while i < len(records):
        r = records[i]
        a, t, v = r.action, r.action.action_type, r.action.value
        who = _who(a.color, me)
        line = None
        if t == ActionType.ROLL:
            total = sum(r.result)
            if total == 7:
                line = f"{who} rolled 7"
            else:
                got = [f"{_who(c, me)} got {freqdeck_to_text(d)}"
                       for c in state.colors
                       if (d := talk.moved.get(i, {}).get(c.value))]
                line = f"{who} rolled {total}: {'; '.join(got) if got else 'nobody produced'}"
        elif t == ActionType.DISCARD_RESOURCE:
            # One engine action per card: gather the run into one line.
            cards = [0] * 5
            while (i < len(records) and records[i].action.color == a.color
                   and records[i].action.action_type == ActionType.DISCARD_RESOURCE):
                cards[RESOURCES.index(records[i].action.value)] += 1
                i += 1
            lines.append(f"  {who} discarded {freqdeck_to_text(cards)}")
            continue
        elif t == ActionType.MOVE_ROBBER:
            coord, victim = v
            if victim is None:
                line = f"{who} moved the robber to {coord}, stealing from nobody"
            elif me in (a.color, victim) and r.result:
                line = (f"{who} moved the robber to {coord} and stole 1 "
                        f"{str(r.result).lower()} from {_who(victim, me)}")
            else:
                line = f"{who} moved the robber to {coord} and stole a card from {_who(victim, me)}"
        elif t == ActionType.BUY_DEVELOPMENT_CARD:
            what = f"a {str(r.result).lower()}" if a.color == me and r.result else "a development card"
            line = f"{who} bought {what}"
        elif t == ActionType.PLAY_MONOPOLY:
            took = talk.moved.get(i, {}).get(a.color.value)
            line = (f"{who} played monopoly on {str(v).lower()}"
                    + (f", taking {freqdeck_to_text(took)}" if took else ""))
        elif t == ActionType.CONFIRM_TRADE:
            line = (f"{who} traded {freqdeck_to_text(v[:5])} to {_who(v[10], me)} "
                    f"for {freqdeck_to_text(v[5:10])}")
        elif t in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY, ActionType.BUILD_ROAD,
                   ActionType.MARITIME_TRADE, ActionType.PLAY_KNIGHT_CARD,
                   ActionType.PLAY_YEAR_OF_PLENTY, ActionType.PLAY_ROAD_BUILDING):
            desc = describe_action(a)
            line = f"{who} " + (desc.replace("build ", "built ", 1).replace("upgrade ", "upgraded ", 1)
                                .replace("trade ", "traded ", 1).replace("play ", "played ", 1))
        if line:
            lines.append(f"  {line}")
        i += 1
    if not lines:
        return "  (nothing)"
    if len(lines) > MAX_EVENTS_SHOWN:
        cut = len(lines) - MAX_EVENTS_SHOWN
        lines = [f"  ({cut} earlier events not shown)"] + lines[cut:]
    return "\n".join(lines)


def render_bank(state) -> str:
    """Public at a table: the piles are in the middle, and the deck's thickness."""
    return (f"  resources: {freqdeck_to_text(state.resource_freqdeck)} | "
            f"development cards left: {len(state.development_listdeck)}")


def render_players(state, me) -> str:
    lines = []
    lr = get_longest_road_color(state)
    la, _ = get_largest_army(state)
    for color in state.colors:
        tags = []
        if color == lr:
            tags.append("longest road")
        if color == la:
            tags.append("largest army")
        if color == me:
            tags.append("you")
        settlements = len(get_player_buildings(state, color, "SETTLEMENT"))
        cities = len(get_player_buildings(state, color, "CITY"))
        roads = len(get_player_buildings(state, color, "ROAD"))
        if color == me:
            hand = freqdeck_to_text(get_player_freqdeck(state, color))
            devs = ", ".join(
                f"{get_dev_cards_in_hand(state, color, d)} {d.lower()}"
                for d in ["KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"]
                if get_dev_cards_in_hand(state, color, d)
            ) or "none"
            cards = f"hand: {hand} | dev cards: {devs}"
            vp = get_actual_victory_points(state, color)
            hidden = vp - get_visible_victory_points(state, color)
            # Say outright that the total already counts the card. Shown a bare
            # "8 VP" next to "1 victory_point", a model read it as 8 visible plus
            # one hidden and planned around a ninth point it did not have.
            vp_txt = f"{vp} VP"
            if hidden:
                vp_txt += (f" ({vp - hidden} that everyone can see + {hidden} from "
                           f"victory point cards only you can see)")
        else:
            # opponents: card COUNTS only, never contents
            cards = (
                f"hand: {player_num_resource_cards(state, color)} cards | "
                f"dev cards: {player_num_dev_cards(state, color)}"
            )
            vp = get_visible_victory_points(state, color)
            vp_txt = f"{vp} VP"
        # Both are public at a real table: the roads are on the board and played
        # knights lie face up. Without the length, a model judged roads only as
        # access to settlement spots and twice passed on the road that would have
        # taken longest road and the game.
        progress = (f"longest continuous road {get_longest_road_length(state, color)}, "
                    f"{get_played_dev_cards(state, color, 'KNIGHT')} knights played")
        ports = sorted(f"2:1 {p.lower()}" if p else "3:1"
                       for p in state.board.get_player_port_resources(color))
        port_txt = f" | ports: {', '.join(ports)}" if ports else ""
        tag = f" ({', '.join(tags)})" if tags else ""
        lines.append(
            f"  {color.value}{tag}: {vp_txt}, {settlements} settlements, "
            f"{cities} cities, {roads} roads ({progress}){port_txt}\n    {cards}"
        )
    return "\n".join(lines)


def describe_action(action) -> str:
    """One legal action, in words rather than tuples."""
    t, v = action.action_type, action.value
    if t == ActionType.BUILD_SETTLEMENT:
        return f"build a settlement at node {v}"
    if t == ActionType.BUILD_CITY:
        return f"upgrade node {v} to a city"
    if t == ActionType.BUILD_ROAD:
        return f"build a road on edge {v}"
    if t == ActionType.BUY_DEVELOPMENT_CARD:
        return "buy a development card"
    if t == ActionType.PLAY_KNIGHT_CARD:
        return "play a knight"
    if t == ActionType.PLAY_MONOPOLY:
        return f"play monopoly on {str(v).lower()}"
    if t == ActionType.PLAY_YEAR_OF_PLENTY:
        return f"play year of plenty for {', '.join(str(x).lower() for x in v)}"
    if t == ActionType.PLAY_ROAD_BUILDING:
        return "play road building"
    if t == ActionType.MOVE_ROBBER:
        coord, victim = v
        who = f", stealing from {victim.value}" if victim else ", stealing from nobody"
        return f"move the robber to {coord}{who}"
    if t == ActionType.MARITIME_TRADE:
        given = [r for r in v[:4] if r is not None]
        return f"trade {len(given)} {str(given[0]).lower()} to the bank for 1 {str(v[4]).lower()}"
    if t == ActionType.OFFER_TRADE:
        return (
            f"offer {freqdeck_to_text(v[:5])} for {freqdeck_to_text(v[5:10])}"
        )
    if t == ActionType.ACCEPT_TRADE:
        return "accept the offer on the table"
    if t == ActionType.REJECT_TRADE:
        return "reject the offer on the table"
    if t == ActionType.CONFIRM_TRADE:
        return f"close the trade with {v[10].value}"
    if t == ActionType.CANCEL_TRADE:
        return "withdraw the offer, nobody trades"
    if t == ActionType.END_TURN:
        return "end your turn"
    if t == ActionType.ROLL:
        return "roll the dice"
    if t == ActionType.DISCARD_RESOURCE:
        return f"discard 1 {str(v).lower()}"
    return f"{t.value} {v}"


# ------------------------------------------------------- where the moves land
#
# The board section names tiles by cube coordinate; the moves name nodes and
# edges by integer id. Nothing in the prompt ever joined the two, so an agent
# choosing an opening settlement was picking between 54 interchangeable
# integers. The first real match shows what that costs: the agents' own
# reasoning hedges ("node 1 *appears* to be a strong intersection", "I'll take
# the first available"), one of them justified node 17 and built on node 20, and
# a value function with a blind opening and perfect play thereafter still loses
# 82 of 100 to the same bot. The annotations below close that gap; everything in
# them is public information the spectator can already see on the board.

#: Ways to roll each number with two dice. The standard measure of how much a
#: node produces — and unusable by an agent that cannot tell which tiles a node
#: touches, which is the whole reason this section exists.
PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


def ports_by_node(state) -> dict:
    """node id -> '2:1 wheat' / '3:1'. `port_nodes` keys the generic port None."""
    return {
        node: f"2:1 {resource.lower()}" if resource else "3:1"
        for resource, nodes in state.board.map.port_nodes.items()
        for node in nodes
    }


def node_neighbours(state) -> dict:
    """node id -> the nodes one edge away, built from the tiles' own edges.

    Needed for the distance rule: a road that reaches a node next to somebody
    else's settlement reaches somewhere you will never be allowed to build.
    """
    adjacency = {}
    for tile in state.board.map.land_tiles.values():
        for a, b in tile.edges.values():
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
    return adjacency


def describe_node(state, node_id, ports=None) -> str:
    """'11 pips: brick 11, brick 6, ore 5' — what a settlement here would touch.

    The robber is marked inline because it sits on a *coordinate* in the board
    section, which an agent cannot match to a node id any better than it can
    match anything else. Pips deliberately ignore it: they are the node's
    standing production, and the robber moves.
    """
    ports = ports if ports is not None else ports_by_node(state)
    robber_tile = state.board.map.tiles.get(state.board.robber_coordinate)
    parts, pips = [], 0
    for tile in state.board.map.adjacent_tiles[node_id]:
        if tile.resource is None:
            parts.append("desert [robber]" if tile is robber_tile else "desert")
            continue
        mark = " [robber]" if tile is robber_tile else ""
        parts.append(f"{tile.resource.lower()} {tile.number}{mark}")
        pips += PIPS.get(tile.number, 0)
    port = ports.get(node_id)
    tail = f", {port} port" if port else ""
    return f"{pips} pips: {', '.join(parts)}{tail}"


def reached_nodes(state, color) -> set:
    """Everywhere this player's network already touches."""
    nodes = set()
    for edge in get_player_buildings(state, color, "ROAD"):
        nodes.update(edge)
    for kind in ("SETTLEMENT", "CITY"):
        nodes.update(get_player_buildings(state, color, kind))
    return nodes


def annotate_action(state, action, me, ports=None, neighbours=None) -> str:
    """`describe_action` plus what the move is worth on this board.

    Only the three moves whose value is positional are annotated. The event log
    keeps the bare description: the replay shows a board, so it does not need
    the geometry spelled out.
    """
    base = describe_action(action)
    t, v = action.action_type, action.value
    if t in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
        return f"{base} ({describe_node(state, v, ports)})"
    if t == ActionType.BUILD_ROAD:
        neighbours = neighbours if neighbours is not None else node_neighbours(state)
        already = reached_nodes(state, me)
        # A legal road always touches the network, so the far end is the one
        # that is new. Both ends are shown for the degenerate case where neither
        # is, rather than silently describing nothing.
        far = [n for n in v if n not in already] or list(v)
        legs = []
        for node in far:
            blocked = node in state.board.buildings or any(
                other in state.board.buildings for other in neighbours.get(node, ())
            )
            # A building is never removed, so a node blocked now stays blocked:
            # this is a permanent fact about the road, not a snapshot.
            rule = ", where nobody may ever settle: too close to a building" if blocked else ""
            legs.append(f"node {node} ({describe_node(state, node, ports)}){rule}")
        return f"{base}, reaching {'; '.join(legs)}"
    return base


def render_actions(state, playable_actions, me) -> str:
    # Built once per prompt rather than per move: a turn can offer 30 roads.
    ports = ports_by_node(state)
    neighbours = node_neighbours(state)
    return "\n".join(
        f"  [{i}] {annotate_action(state, a, me, ports, neighbours)}"
        for i, a in enumerate(playable_actions)
    )


def render_offer_on_table(state) -> str:
    trade = state.current_trade
    if trade is None:
        return ""
    proposer = state.colors[trade[10]]
    return (
        f"{proposer.value} is offering you {freqdeck_to_text(trade[:5])} "
        f"in exchange for {freqdeck_to_text(trade[5:10])}."
    )


def takeable_counters(state, talk, me) -> list:
    """Counter-offers made to `me` this turn that it can still take up: both
    sides still hold their cards, and it has an offer left to spend on it.
    Taking one up puts the counter's terms back to its author alone."""
    if talk.offers_left(state.num_turns, me.value) <= 0:
        return []
    mine = get_player_freqdeck(state, me)
    out = []
    for _, c in talk.counters_to(state.num_turns, me.value):
        theirs = get_player_freqdeck(state, next(x for x in state.colors if x.value == c.color))
        if (all(h >= w for h, w in zip(mine, c.want))
                and all(h >= g for h, g in zip(theirs, c.give))):
            out.append(c)
    return out


def describe_counter(counter) -> str:
    return (f"take up {counter.color}'s counter-offer: you give "
            f"{freqdeck_to_text(counter.want)}, you get {freqdeck_to_text(counter.give)} "
            f"(put to {counter.color} alone; it still has to accept)")


def render_discard(state, me) -> str:
    left = state.discard_counts[state.color_to_index[me]]
    return (f"A 7 was rolled and you hold more than {state.discard_limit} cards. "
            f"You must discard {left} more card(s), one per move.")


def build_prompt(game, me, playable_actions, talk, may_offer: bool, since: int = 0,
                 note=None, may_counter: bool = False) -> str:
    """`since`: the index into `state.action_records` at this seat's previous
    prompt. `note`: (turn, reasoning) from its last decision on its own turn —
    a player remembers what it was trying to do; a model call starts from
    nothing. `may_counter`: the seat is answering an offer and may name other
    terms instead."""
    state = game.state
    sections = [
        f"TURN {state.num_turns} — you are {me.value}.",
        f"\nBOARD (robber at {state.board.robber_coordinate}):\n{render_board(state)}",
        f"\nROADS (each an edge between two nodes):\n{render_roads(state, me)}",
        f"\nPORTS (a settlement or city on either node may use it):\n{render_ports(state)}",
        f"\nPLAYERS:\n{render_players(state, me)}",
        f"\nBANK:\n{render_bank(state)}",
        f"\nSINCE YOUR LAST DECISION, oldest first:\n{render_history(state, me, since, talk)}",
        f"\nTABLE TALK (recent):\n{talk.transcript(state.num_turns)}",
    ]
    if note:
        sections.append(f"\nYOUR OWN NOTE from your last move on your own turn "
                        f"(turn {note[0]}), which nobody else saw:\n  {note[1]}")
    if state.current_prompt == ActionPrompt.DECIDE_TRADE:
        sections.append(f"\nON THE TABLE: {render_offer_on_table(state)}")
    if state.current_prompt == ActionPrompt.DISCARD:
        sections.append(f"\nDISCARD: {render_discard(state, me)}")
    counters = takeable_counters(state, talk, me) if may_offer else []
    moves = render_actions(state, playable_actions, me)
    if counters:
        n = len(playable_actions)
        moves += "\n" + "\n".join(f"  [{n + k}] {describe_counter(c)}"
                                   for k, c in enumerate(counters))
    sections.append(
        f"\nYOUR LEGAL MOVES (pips = ways to roll that number out of 36, summed "
        f"over the tiles a node touches; 6 and 8 are 5 pips each, 2 and 12 are 1):"
        f"\n{moves}"
    )
    if may_offer:
        left = talk.offers_left(state.num_turns, me.value)
        sections.append(
            f"\nYou may instead write your own trade offer to the table: set choice to "
            f"{AUTHOR_OFFER}, fill offer_give and offer_want with resource names "
            f"(wood, brick, sheep, wheat, ore — repeat a name to offer more than one), "
            f"and put your pitch in `say`. You can only offer cards you actually hold. "
            f"You have {left} offer(s) left this turn"
            + (", and taking up a counter-offer uses one." if counters else ".")
        )
    if may_counter:
        proposer = state.colors[state.current_trade[10]].value
        sections.append(
            f"\nYou may instead counter with your own terms: set choice to {AUTHOR_OFFER}, "
            f"fill offer_give with what you would give {proposer} and offer_want with what "
            f"you want from {proposer}, and put your pitch in `say`. You can only offer "
            f"cards you actually hold. It counts as turning this offer down; {proposer} "
            f"may then put your terms to you alone, and you will be asked again."
        )
    sections.append(
        "\nAnswer with your note, your choice, and — if you are offering or "
        "responding to an offer — what you say out loud in `say`. Everyone at the "
        "table hears `say`. Nothing you say binds you."
    )
    return "\n".join(sections)


# ---------------------------------------------------------------- reading it back


def resources_to_freqdeck(names: List[str]) -> Tuple[int, ...]:
    """['wood','wood','ore'] -> (2,0,0,0,1). Unknown names are dropped."""
    deck = [0] * 5
    lookup = {r.lower(): i for i, r in enumerate(RESOURCES)}
    for n in names:
        i = lookup.get(str(n).strip().lower())
        if i is not None:
            deck[i] += 1
    return tuple(deck)


def validate_offer(state, me, give: Tuple[int, ...], want: Tuple[int, ...]) -> Optional[str]:
    """Return a complaint if the offer is not one this player could make."""
    if sum(give) == 0 or sum(want) == 0:
        return "an offer must have something on both sides"
    hand = get_player_freqdeck(state, me)
    short = [
        f"{g - h} more {r.lower()}"
        for g, h, r in zip(give, hand, RESOURCES)
        if g > h
    ]
    if short:
        return f"you do not hold what you offered — you would need {', '.join(short)}"
    return None

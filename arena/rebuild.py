"""Put a finished match back into the engine, position by position.

`arena/replay.py` turns a run file into frames for a spectator; this turns it
back into a live `Game`, so any position an agent faced can be examined with the
engine's own functions: what was legal, what the bot would have played, whether
a win was on the board. It is what found that an agent twice passed on the road
that would have won it the game.

It works because the run file carries everything the engine drew at random: the
seed rebuilds the same board and the same seating, and every `last_action` holds
its result — the dice, the stolen card, the development card drawn — which
`apply_action` takes instead of consulting the rng. Nothing is inferred.

Every ply is checked against the snapshot the match wrote at the time, and any
difference raises. A reconstruction that drifted would still produce plausible
positions, and every conclusion drawn from them would be wrong.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union

import arena  # noqa: F401  registers the extra seat colours before seating anyone
from catanatron.game import Game
from catanatron.models.enums import Action
from catanatron.models.player import Color, RandomPlayer
from catanatron.serialization import action_record_from_json

from arena.events import read_run
from arena.llm_player import OWN_TURN_PROMPTS, may_counter, may_offer, stable_order
from arena.personas import ROSTER
from arena.prompt import build_prompt
from arena.table_talk import TableTalk

#: Event kinds that say who acted on a ply and how the action was chosen.
ACTOR_KINDS = {"decision", "reflex", "bot", "own_offer_skipped", "not_addressed",
               "fallback"}


class RebuildError(Exception):
    pass


@dataclass
class Ply:
    """One action, with the position it was chosen in.

    `game` and `talk` are live: the position *before* `action` is applied,
    advanced as soon as iteration continues. Take `game.copy()` to keep a
    position or to try moves in it — never mutate either.
    """
    index: int
    game: Game
    action: Action
    #: the decision/reflex/bot event for this ply, with reasoning and say when a
    #: model chose it; None if the log has no actor event for it
    actor: Optional[dict]
    #: the state event the match wrote after this ply
    snapshot: dict
    #: the run's game_start event
    start: dict
    #: the table talk as it stood when the move was chosen; advanced with `game`
    talk: TableTalk = None
    #: where `state.action_records` stood at this seat's previous prompt
    since: int = 0
    #: (turn, reasoning) of this seat's last decision on its own turn
    note: Optional[tuple] = None
    #: `since` and `note` for every seat, not just the one acting: what a
    #: resumed match hands each LLMPlayer
    seen_all: Optional[dict] = None
    notes_all: Optional[dict] = None

    @property
    def seat(self) -> dict:
        return next(s for s in self.start["seats"] if s["color"] == self.action.color.value)

    @property
    def chosen_by(self) -> Optional[str]:
        """'decision' when a model chose, 'reflex', 'bot', ... otherwise."""
        return self.actor["kind"] if self.actor else None

    def legal(self) -> list:
        """The legal moves in the order the agent was shown them."""
        return stable_order(self.game.playable_actions)

    def prompt(self) -> tuple:
        """(system, user): what an agent in this seat was sent, built by the same
        code that built it during the match. Only the first attempt — a retry
        also carried the complaint about the reply before it."""
        color = self.action.color
        legal = self.legal()
        persona = ROSTER[self.seat["name"]]
        user = build_prompt(self.game, color, legal, self.talk,
                            may_offer(self.game.state, legal, self.talk, color),
                            since=self.since, note=self.note,
                            may_counter=may_counter(self.game.state, color))
        return persona.system_prompt, user


def plies(run: Union[str, Path, List[dict]]) -> Iterator[Ply]:
    """Every ply of a match, in order, each verified against its snapshot."""
    events = read_run(run) if isinstance(run, (str, Path)) else run
    if not events or events[0]["kind"] != "game_start":
        raise RebuildError("not a run file: no game_start")
    start = events[0]
    # Seats in the order they were constructed; the engine then shuffles them
    # with the seed, exactly as it did in the match.
    game = Game(players=[RandomPlayer(Color[s["color"]]) for s in start["seats"]],
                seed=start["seed"])
    if [c.value for c in game.state.colors] != start["order"]:
        raise RebuildError(
            f"seating differs: rebuilt {[c.value for c in game.state.colors]}, "
            f"recorded {start['order']} — different engine version?")

    talk = TableTalk(max_offers_per_turn=start.get("max_offers_per_turn", 3))
    actor, index = None, 0
    seen = {}  # color -> action_records length at its last prompt, as LLMPlayer keeps it
    notes = {}  # color -> (turn, reasoning), as LLMPlayer keeps it
    talk.observe(game.state)
    for event in events[1:]:
        if event["kind"] in ACTOR_KINDS:
            actor = event
            continue
        if event["kind"] != "state":
            continue
        last = event["last_action"]
        record = action_record_from_json(
            [[last["color"], last["type"], last["value"]], last["result"]])
        # Every ply in the real matches has its own actor event; the check is so a
        # ply without one gets None rather than a neighbour's reasoning.
        mine = actor if actor and actor.get("color") == last["color"] else None
        yield Ply(index, game, record.action, mine, event, start, talk,
                  seen.get(last["color"], 0), notes.get(last["color"]),
                  dict(seen), dict(notes))

        if mine and mine["kind"] in ("decision", "fallback"):
            # Both are emitted only after LLMPlayer built a prompt, which is
            # exactly where it moves its own marker.
            seen[last["color"]] = len(game.state.action_records)

        if mine and mine["kind"] == "decision":
            # Only a model's move carries speech; the player recorded it with the
            # turn as it stood before the move, which is now.
            talk.record_move(game.state.num_turns, last["color"], record.action,
                             mine.get("say"), to=mine.get("to"), counter=mine.get("counter"))
            if game.state.current_prompt in OWN_TURN_PROMPTS:
                notes[last["color"]] = (game.state.num_turns, mine.get("reasoning"))
        game.execute(record.action, validate_action=False, action_record=record)
        talk.observe(game.state)
        if dict(game.state.player_state) != event["player_state"]:
            differing = sorted(k for k, v in event["player_state"].items()
                               if game.state.player_state.get(k) != v)
            raise RebuildError(
                f"ply {index} ({last['color']} {last['type']}, seq {event['seq']}): "
                f"rebuilt state differs from the snapshot in {differing[:6]}")
        actor, index = None, index + 1


def position(run, predicate) -> Optional[Ply]:
    """The first ply matching `predicate`, copied so it outlives the iteration
    and can be played forward."""
    for ply in plies(run):
        if predicate(ply):
            return detach(ply)
    return None


def detach(ply: Ply) -> Ply:
    """A copy of `ply` that iteration will not advance underneath you."""
    return Ply(ply.index, ply.game.copy(), ply.action, ply.actor, ply.snapshot,
               ply.start, ply.talk.copy(), ply.since, ply.note,
               dict(ply.seen_all or {}), dict(ply.notes_all or {}))


def resume_point(run, turn: Optional[int] = None) -> Ply:
    """Where an interrupted match picks up: the roll that opens `turn`, or the
    last roll the log holds. A turn start, because mid-turn the table holds
    offers and answers in flight that no event fully describes.

    Reads past a torn tail: a crash can leave the last line half written or
    padded with zero bytes, and everything before it is still good."""
    if isinstance(run, (str, Path)):
        from arena.events import read_run_lenient
        run = read_run_lenient(run)
    found = None
    for ply in plies(run):
        if ply.action.action_type.value != "ROLL":
            continue
        if turn is None or ply.game.state.num_turns == turn:
            found = detach(ply)
            if turn is not None:
                break
    if found is None:
        raise RebuildError(f"no roll opening turn {turn}")
    return found

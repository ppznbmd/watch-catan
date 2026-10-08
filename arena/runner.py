"""Run one match and narrate it.

Drives the engine a ply at a time rather than calling `Game.play()`, because the
observer hook fires *before* an action is applied and a spectator wants the board
as it is *after*.
"""

import json
import time
import uuid
from typing import List, Optional

from catanatron.game import TURNS_LIMIT, Game
from catanatron.observer import GameObserver
from catanatron.serialization import action_record_to_json, geometry, web_view
from catanatron.models.player import Color
from catanatron.state_functions import get_actual_victory_points

from arena.bot import TradingValuePlayer
from arena.events import EventLog
from arena.llm_player import LLMPlayer
from arena.prompt import PROMPT_VERSION, describe_action
from arena.table_talk import TableTalk


class BotNarrator(GameObserver):
    """Log what the non-LLM seats do.

    Only LLMPlayer writes its own events, so without this a replay shows a table
    where the baseline bot never acts — its moves are real and in the engine, just
    unrecorded. The engine calls `step` for every action by anyone, which is the
    one place that sees them all.
    """

    def __init__(self, log, bot_colors):
        self.log = log
        self.bots = set(bot_colors)

    def step(self, game_before_action, action):
        if action.color not in self.bots:
            return  # the LLM seats narrate themselves, with their reasoning
        self.log.emit(
            "bot",
            color=action.color.value,
            action={
                "type": action.action_type.value,
                "value": action.value,
                "described": describe_action(action),
            },
        )


def encode(game) -> dict:
    """Board geometry — the static half, sent once. The engine derives node and
    edge topology from the map here so the browser does not re-implement it."""
    return json.loads(json.dumps(geometry(game), default=str))


def player_view(game, perspective) -> dict:
    """The web payload as one seat sees it, with every other hand redacted.

    `perspective` is required on purpose. In Catanatron `perspective=None` is the
    *omniscient* view — the engine's own docstring says to leave it out "for a
    spectator, a replay, or an accumulator". That default is right for a replay
    and catastrophic for anything a player can see, so this wrapper does not
    offer it. A UI that deliberately wants the omniscient view calls `web_view`
    itself and says so.
    """
    if perspective is None:
        raise ValueError(
            "player_view needs a seat; web_view(game, perspective=None) is the "
            "omniscient view and must be chosen explicitly"
        )
    return json.loads(json.dumps(web_view(game, perspective=perspective), default=str))


def last_action(state) -> Optional[dict]:
    """The action just applied, and what the engine rolled, drew or stole to carry
    it out.

    The decision events say what a seat *chose*; only the engine knows what came of
    it. Without this a replay shows the robber moving with no 7 ever rolled, and a
    development card appearing from nowhere. Catanatron keeps the outcome on the
    `ActionRecord`, which is the only place it exists — `Action` itself does not
    carry it.

    Catanatron redacts `BUY_DEVELOPMENT_CARD` and `MOVE_ROBBER` results from other
    seats (`serialization.SECRET_RESULT`). Nothing is redacted here: `runs/` is the
    authoritative record, and the redaction belongs on the way *to* an agent, which
    is `arena/prompt.py`. Never feed a run file back to a player.
    """
    if not state.action_records:
        return None
    record = state.action_records[-1]
    (color, action_type, value), result = action_record_to_json(record)
    return {
        "color": color,
        "type": action_type,
        "value": value,
        "described": describe_action(record.action),
        "result": result,
    }


def snapshot(game) -> dict:
    """Only what changes: buildings, roads, robber, per-player counters.

    This is **omniscient on purpose**: `player_state` carries every hand exactly,
    because the JSONL is the replay and the dataset, and analysis needs to see
    what each agent was actually holding when it bluffed. No agent ever reads it —
    agents only ever see the rendered prompt. A player-facing UI must not serve
    this raw; use `player_view` for that.
    """
    state = game.state
    return {
        "last_action": last_action(state),
        "turn": state.num_turns,
        "current_color": state.current_color().value,
        "prompt": state.current_prompt.value,
        "robber": list(state.board.robber_coordinate),
        "buildings": {
            str(node): [b[0].value, b[1]] for node, b in state.board.buildings.items()
        },
        "roads": {
            str(sorted(edge)): color.value for edge, color in state.board.roads.items()
        },
        "player_state": dict(state.player_state),
        "victory_points": {
            c.value: get_actual_victory_points(state, c) for c in state.colors
        },
    }


def run_match(
    seats: List[tuple],
    game_id: Optional[str] = None,
    max_offers_per_turn: int = 2,
    runs_dir=None,
    seed: Optional[int] = None,
    on_event=None,
    scratch: bool = False,
    max_turns: int = TURNS_LIMIT,
    resume=None,
    resume_seed: Optional[int] = None,
) -> dict:
    """`seats` is a list of (Color, player_factory). A factory takes
    (color, talk, log) and returns a Player — so an LLM seat and a baseline bot
    seat are built the same way.

    `max_turns` ends a match with no winner. The engine's own limit is 1000
    turns, ten times the longest match played here; a seat that plays to stop
    others winning can stretch a match toward it, and every turn is paid for.

    `resume` is a `Ply` from `arena.rebuild.resume_point`: the match continues
    from that position in a new run file whose game_start names the run and ply
    it came from. Each LLM seat gets back what it would have seen next (its last
    note, how much history it had read); the agents keep no other memory. The
    dice from there on come from `resume_seed`, since the rebuilt game never
    drew from its rng and would otherwise replay the opening rolls."""
    game_id = game_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    talk = resume.talk if resume else TableTalk(max_offers_per_turn=max_offers_per_turn)

    with EventLog(game_id, runs_dir=runs_dir, scratch=scratch) as log:
        if on_event:
            log.subscribe(on_event)
        players = [factory(color, talk, log) for color, factory in seats]
        if resume:
            game = resume.game
            by_color = {p.color: p for p in players}
            if set(by_color) != set(game.state.colors):
                raise ValueError(f"seats {sorted(c.value for c in by_color)} do not match "
                                 f"the resumed table {[c.value for c in game.state.colors]}")
            game.state.players = [by_color[c] for c in game.state.colors]
            for p in players:
                if isinstance(p, LLMPlayer):
                    p._seen = resume.seen_all.get(p.color.value, 0)
                    p._note = resume.notes_all.get(p.color.value)
            game.random.seed(resume_seed)
        else:
            game = Game(players=players, seed=seed)
        game.watch(BotNarrator(
            log, [p.color for p in players if not isinstance(p, LLMPlayer)]
        ))

        log.emit(
            "game_start",
            game_id=game_id,
            seed=game.seed,
            seats=[
                {
                    "color": p.color.value,
                    "kind": "llm" if isinstance(p, LLMPlayer) else "bot",
                    "name": getattr(getattr(p, "persona", None), "name", type(p).__name__),
                    "blurb": getattr(getattr(p, "persona", None), "blurb", ""),
                    "model": getattr(p, "model", None),
                    # Effort changes how well a model plays (the win probe: Luna
                    # finds 54% of wins at 'high', 38% at its default), so
                    # matches at different efforts are different groups.
                    "effort": getattr(getattr(p, "decider", None), "effort", None),
                    # The tier asked for. Each call's usage records the tier
                    # actually served, which is what the bill follows.
                    "service_tier": getattr(getattr(p, "decider", None), "service_tier", None),
                }
                for p in players
            ],
            # `player_state` is keyed P0..P3 by seating index, and `State.__init__`
            # shuffles the seats — so without this, nothing downstream can say whose
            # hand is whose. A replay that guesses would attribute cards to the wrong
            # player and look entirely plausible doing it.
            order=[c.value for c in game.state.colors],
            # The offer budget changes what a prompt says ("N offer(s) left"), so a
            # rebuilt prompt needs it. Matches before this was logged used 3.
            max_offers_per_turn=max_offers_per_turn,
            max_turns=max_turns,
            # What the agents were shown. Runs before version 3 do not carry it.
            prompt_version=PROMPT_VERSION,
            board=encode(game),
            **({"resumed_from": {"run": resume.start["game_id"], "ply": resume.index,
                                 "turn": resume.game.state.num_turns,
                                 "resume_seed": resume_seed}} if resume else {}),
        )
        if not resume:
            talk.observe(game.state)

        started = time.time()
        while game.winning_color() is None and game.state.num_turns < max_turns:
            game.play_tick()
            talk.observe(game.state)
            log.emit("state", **snapshot(game))

        winner = game.winning_color()
        summary = {
            "game_id": game_id,
            "winner": winner.value if winner else None,
            "truncated": winner is None,
            "turns": game.state.num_turns,
            "actions": len(game.state.action_records),
            "seconds": round(time.time() - started, 1),
            "events": log.count,
            "offers": sum(1 for u in talk.utterances if u.kind == "pitch"),
            "agents": {
                p.color.value: {
                    "name": getattr(getattr(p, "persona", None), "name", type(p).__name__),
                    "model": getattr(p, "model", None),
                    "victory_points": get_actual_victory_points(game.state, p.color),
                    "llm_calls": getattr(p, "calls", 0),
                    "fallbacks": getattr(p, "fallbacks", 0),
                    "malformed": getattr(p, "malformed", 0),
                    "self_offers_skipped": getattr(p, "self_offers", 0),
                    "usage": getattr(p, "usage", {}),
                }
                for p in players
            },
            "path": str(log.path),
        }
        log.emit("game_over", **summary)
    return summary


def llm_seat(persona, decider):
    """A factory for an LLM seat with a given persona."""

    def factory(color, talk, log):
        return LLMPlayer(color, persona, decider, talk, log)

    return factory


def bot_seat(cls, *args, **kwargs):
    """A factory for a bot seat — the baseline ruler. A bot that answers trade
    offers is handed the table talk, to see who an offer is put to."""

    def factory(color, talk, log):
        if issubclass(cls, TradingValuePlayer):
            return cls(color, talk, *args, **kwargs)
        return cls(color, *args, **kwargs)

    return factory

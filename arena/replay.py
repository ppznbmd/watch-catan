"""Read a match back: finished, or still being played.

`runs/*.jsonl` is a stream of events, not a sequence of positions. This turns it
into frames — one board position per ply, with whatever led to it attached — which
is the shape a spectator needs and the shape an analysis wants.

Reading is incremental and the same code serves both cases: a finished file is
polled once, a live file is polled repeatedly and yields whatever is new. That is
why there is no separate live path and no SSE server. `EventLog` opens its file
line-buffered precisely so this works.

The one thing a reader may not do is guess. `player_state` is keyed `P0..P3` by
seating index and `State.__init__` shuffles the seats, so the colour of each seat
comes from the `order` recorded at `game_start` and from nowhere else.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEV_CARDS = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")

#: An action that is also a speech act, and what it means at the table. A
#: rejection that names other terms is a "counter", decided from the event.
TALK_KIND = {
    "OFFER_TRADE": "pitch",
    "ACCEPT_TRADE": "reply",
    "REJECT_TRADE": "reply",
    "CONFIRM_TRADE": "confirm",
    "CANCEL_TRADE": "cancel",
}

#: Event kinds that mean a seat did not get what the model was asked for. They are
#: separated out because a match degrades silently otherwise: the reflex policy
#: fills every gap and the result still looks like a complete, plausible game.
DEGRADED = {"malformed", "invalid", "fallback", "error"}


class ReplayError(Exception):
    pass


class Replay:
    """Frames rebuilt from a run file.

    Construct, then `poll()` — repeatedly, if the match is still being played.
    Each call returns only the frames that are new, so a caller can stream them.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.meta: Optional[dict] = None
        self.summary: Optional[dict] = None
        self.frames: List[dict] = []
        self._offset = 0
        self._partial = b""
        self._pending: List[dict] = []

    @property
    def complete(self) -> bool:
        """A match that reached `game_over`. A live one has not, and neither has
        one whose process was killed — the two are indistinguishable from here."""
        return self.summary is not None

    @property
    def colors(self) -> List[str]:
        """Seats in turn order, which is also `P0..P3` order."""
        return list(self.meta["order"]) if self.meta else []

    def poll(self) -> List[dict]:
        """Read whatever has been appended since the last call.

        A live file may end mid-line: the writer's flush and our read are not
        synchronised. An incomplete tail is held back and completed on a later
        poll rather than being parsed into a half-event.
        """
        if not self.path.exists():
            return []
        with self.path.open("rb") as fh:
            fh.seek(self._offset)
            chunk = fh.read()
        if not chunk:
            return []
        self._offset += len(chunk)

        lines = (self._partial + chunk).split(b"\n")
        self._partial = lines.pop()  # empty when the chunk ended on a newline
        first_new = len(self.frames)
        for line in lines:
            if line.strip():
                self._ingest(json.loads(line))
        return self.frames[first_new:]

    def read_all(self) -> List[dict]:
        """Every frame in a file, for a caller that does not care about streaming."""
        self.poll()
        return self.frames

    # ---------------------------------------------------------------- internals

    def _ingest(self, event: dict) -> None:
        kind = event["kind"]
        if kind == "game_start":
            self._start(event)
        elif kind == "state":
            self.frames.append(self._frame(event))
            self._pending = []
        elif kind == "game_over":
            self.summary = event
        else:
            self._pending.append(event)

    def _start(self, event: dict) -> None:
        if "order" not in event:
            raise ReplayError(
                f"{self.path.name} was written before the seating order was logged, "
                "so no hand can be attributed to a seat. Play the match again."
            )
        self.meta = event
        # The position before anyone moved. The log has no event for it — it is the
        # board as handed out — but a spectator should be able to start there. The
        # robber is already on the desert at that point (Board.__init__ puts it on
        # the first land tile with no resource), so it is read off the recorded
        # board rather than left out, which would say there is no robber.
        self.frames.append(
            {
                "seq": -1,
                "turn": 0,
                "current_color": event["order"][0],
                "prompt": "BUILD_INITIAL_SETTLEMENT",
                "last_action": None,
                "robber": _desert(event.get("board", {})),
                "buildings": {},
                "roads": {},
                "players": {
                    color: _empty_seat() for color in event["order"]
                },
                "events": [],
                "talk": [],
                "degraded": [],
            }
        )

    def _frame(self, event: dict) -> dict:
        state = event["player_state"]
        players = {}
        for index, color in enumerate(self.colors):
            players[color] = _seat(state, index, event.get("victory_points", {}).get(color))
        talk = [_utterance(e) for e in self._pending if _utterance(e)]
        return {
            "seq": event["seq"],
            "turn": event["turn"],
            "current_color": event["current_color"],
            "prompt": event["prompt"],
            "last_action": event.get("last_action"),
            "robber": event["robber"],
            "buildings": event["buildings"],
            "roads": event["roads"],
            "players": players,
            "events": self._pending,
            "talk": talk,
            "degraded": [e for e in self._pending if e["kind"] in DEGRADED],
        }


def _desert(board: dict) -> Optional[list]:
    for tile in board.get("tiles", []):
        if tile.get("tile", {}).get("type") == "DESERT":
            return tile["coordinate"]
    return None


def _empty_seat() -> dict:
    return {
        "victory_points": 0,
        "hand": {r: 0 for r in RESOURCES},
        "cards": {c: 0 for c in DEV_CARDS},
        "played": {c: 0 for c in DEV_CARDS},
        "roads_available": 15,
        "settlements_available": 5,
        "cities_available": 4,
        "longest_road": 0,
        "has_road": False,
        "has_army": False,
    }


def _seat(state: Dict, index: int, victory_points) -> dict:
    """One seat's public counters and its hand, out of the flat `P{i}_` namespace."""
    def value(key, default=0):
        return state.get(f"P{index}_{key}", default)

    return {
        # ACTUAL_VICTORY_POINTS counts the victory-point cards nobody else can see.
        # The snapshot's per-colour tally is the same number; prefer it when present.
        "victory_points": victory_points if victory_points is not None
        else value("ACTUAL_VICTORY_POINTS"),
        "public_victory_points": value("VICTORY_POINTS"),
        "hand": {r: value(f"{r}_IN_HAND") for r in RESOURCES},
        "cards": {c: value(f"{c}_IN_HAND") for c in DEV_CARDS},
        "played": {c: value(f"PLAYED_{c}") for c in DEV_CARDS},
        "roads_available": value("ROADS_AVAILABLE", 15),
        "settlements_available": value("SETTLEMENTS_AVAILABLE", 5),
        "cities_available": value("CITIES_AVAILABLE", 4),
        "longest_road": value("LONGEST_ROAD_LENGTH"),
        "has_road": bool(value("HAS_ROAD", False)),
        "has_army": bool(value("HAS_ARMY", False)),
    }


def _utterance(event: dict) -> Optional[dict]:
    """The public half of a decision, and the private half beside it.

    `say` is what the table heard; `reasoning` is what the agent told itself. The
    gap between them is the point of the project, so a replay keeps both and never
    merges them.
    """
    if event["kind"] != "decision":
        return None
    action = event.get("action") or {}
    kind = TALK_KIND.get(action.get("type"))
    if event.get("counter"):
        kind = "counter"
    said = (event.get("say") or "").strip()
    if not kind and not said:
        return None
    value = action.get("value") or []
    if kind == "counter":
        give, want = event["counter"]
    elif kind == "pitch":
        give, want = list(value[:5]), list(value[5:10])
    else:
        give = want = None
    return {
        "color": event["color"],
        "persona": event.get("persona"),
        "kind": kind or "aside",
        "text": said,
        "described": action.get("described"),
        "reasoning": event.get("reasoning") or "",
        "give": give,
        "want": want,
        # a pitch put to one player alone; None for an offer to the table
        "to": event.get("to"),
    }


def load(path) -> Replay:
    """Read a whole file at once."""
    replay = Replay(path)
    replay.read_all()
    return replay

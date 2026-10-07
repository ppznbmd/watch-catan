"""The negotiation transcript.

The engine moves resources; this moves words. It lives outside engine state and
never affects what is legal, which is what keeps Catanatron unforked.

Everything here is public: every agent at the table sees every pitch and every
reply. That is what makes a reputation — or a bluff — possible at all.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from catanatron.models.enums import RESOURCES, ActionType
from catanatron.state_functions import get_player_freqdeck


def freqdeck_to_text(freqdeck) -> str:
    """(1,0,2,0,0) -> '1 wood, 2 sheep'."""
    parts = [
        f"{n} {res.lower()}" for n, res in zip(freqdeck, RESOURCES) if n
    ]
    return ", ".join(parts) if parts else "nothing"


@dataclass
class Utterance:
    """One thing one player said, attached to one offer."""

    turn: int
    color: str
    kind: str  # "pitch" | "reply" | "counter" | "confirm" | "cancel" | "aside"
    text: str
    give: Optional[tuple] = None  # the speaker's side of the offer, freqdeck
    want: Optional[tuple] = None
    #: A pitch put to one player alone (a counter-offer taken up), or the
    #: proposer a counter answers. Said out loud like everything else here.
    to: Optional[str] = None

    def render(self) -> str:
        if self.kind == "pitch":
            who = f" to {self.to} only" if self.to else ""
            return (
                f"{self.color} offers{who} {freqdeck_to_text(self.give)} "
                f"for {freqdeck_to_text(self.want)} — \"{self.text}\""
            )
        if self.kind == "counter":
            return (
                f"{self.color} counters {self.to}'s offer: would give "
                f"{freqdeck_to_text(self.give)} for {freqdeck_to_text(self.want)} "
                f"— \"{self.text}\""
            )
        if self.kind == "reply":
            return f"{self.color} replies: \"{self.text}\""
        return f"{self.color}: {self.text}"


@dataclass
class TableTalk:
    """Per-game transcript, plus the per-turn offer budget.

    The budget is keyed on the engine's turn counter, not on a flag we reset
    ourselves — a flag only gets cleared in a branch that may never run, and an
    agent then haggles forever.
    """

    max_offers_per_turn: int = 3
    utterances: List[Utterance] = field(default_factory=list)
    _offers: dict = field(default_factory=dict)  # (turn, color) -> count
    #: counters already put back to the table as a directed offer, by position
    #: in `utterances` — a counter is taken up at most once
    _taken: set = field(default_factory=set)
    #: action_records index -> {color: freqdeck gained or lost}, for the public
    #: events whose effect is not in the record itself: what a roll paid out and
    #: what a monopoly took. Everybody at a table watches cards leave the bank.
    moved: Dict[int, Dict[str, tuple]] = field(default_factory=dict)
    #: hands after the last observed action. Private, and never rendered: kept
    #: only to take the difference that `moved` records.
    _hands: Dict[str, tuple] = field(default_factory=dict)
    _observed: int = 0

    def copy(self) -> "TableTalk":
        return TableTalk(self.max_offers_per_turn, list(self.utterances), dict(self._offers),
                         set(self._taken), dict(self.moved), dict(self._hands), self._observed)

    def observe(self, state) -> None:
        """Called after every action, by `arena/runner.py` during a match and by
        `arena/rebuild.py` afterwards. The engine does not record what a roll
        produced, and by the time an agent is next asked the board may have
        changed, so it cannot be recomputed later."""
        hands = {c.value: tuple(get_player_freqdeck(state, c)) for c in state.colors}
        for i in range(self._observed, len(state.action_records)):
            t = state.action_records[i].action.action_type
            if t in (ActionType.ROLL, ActionType.PLAY_MONOPOLY) and self._hands:
                delta = {c: tuple(a - b for a, b in zip(hands[c], self._hands[c]))
                         for c in hands}
                self.moved[i] = {c: d for c, d in delta.items() if any(d)}
        self._hands, self._observed = hands, len(state.action_records)

    def offers_left(self, turn: int, color: str) -> int:
        return self.max_offers_per_turn - self._offers.get((turn, color), 0)

    def record_pitch(self, turn, color, give, want, text, to=None) -> Utterance:
        self._offers[(turn, color)] = self._offers.get((turn, color), 0) + 1
        if to:
            for i, c in self.counters_to(turn, color):
                if c.color == to and (c.give, c.want) == (tuple(want), tuple(give)):
                    self._taken.add(i)
                    break
        u = Utterance(turn, color, "pitch", text, tuple(give), tuple(want), to)
        self.utterances.append(u)
        return u

    def counters_to(self, turn, proposer) -> List[tuple]:
        """(position, counter) for every counter made to `proposer` this turn and
        not yet taken up, oldest first."""
        return [(i, u) for i, u in enumerate(self.utterances)
                if u.turn == turn and u.kind == "counter" and u.to == proposer
                and i not in self._taken]

    def addressee(self) -> Optional[str]:
        """Who the offer on the table is put to, if only one player. The offer on
        the table is always the latest pitch: only LLM seats make offers, and
        each is recorded as it is made."""
        pitch = next((u for u in reversed(self.utterances) if u.kind == "pitch"), None)
        return pitch.to if pitch else None

    def record(self, turn, color, kind, text) -> Utterance:
        u = Utterance(turn, color, kind, text)
        self.utterances.append(u)
        return u

    def record_move(self, turn, color, action, say, to=None, counter=None) -> Optional[Utterance]:
        """Record what a seat said with a move, attached to the right offer.

        The one place talk is recorded, both during a match and when
        `arena/rebuild.py` restores a transcript — so a rebuilt prompt cannot
        hear a table the agent never heard. `to` directs an offer at one player;
        `counter` is (give, want) when a rejection came with other terms.
        """
        said = (say or "").strip()
        t = action.action_type
        if t == ActionType.OFFER_TRADE:
            return self.record_pitch(turn, color, action.value[:5], action.value[5:10], said,
                                     to=to)
        if t == ActionType.REJECT_TRADE and counter:
            proposer = next(u.color for u in reversed(self.utterances) if u.kind == "pitch")
            u = Utterance(turn, color, "counter", said, tuple(counter[0]), tuple(counter[1]),
                          proposer)
            self.utterances.append(u)
            return u
        if t in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE):
            verdict = "accepts" if t == ActionType.ACCEPT_TRADE else "rejects"
            return self.record(turn, color, "reply", said or f"({verdict}, without comment)")
        if t == ActionType.CONFIRM_TRADE:
            return self.record(turn, color, "confirm",
                               said or f"closes the trade with {action.value[10].value}")
        if t == ActionType.CANCEL_TRADE:
            return self.record(turn, color, "cancel", said or "withdraws the offer")
        if said:
            return self.record(turn, color, "aside", said)
        return None

    def this_turn(self, turn: int) -> List[Utterance]:
        return [u for u in self.utterances if u.turn == turn]

    def transcript(self, turn: int, lookback: int = 2) -> str:
        """What an agent is allowed to have heard, most recent turns last."""
        recent = [u for u in self.utterances if u.turn > turn - lookback]
        if not recent:
            return "(nothing said yet)"
        return "\n".join(f"  turn {u.turn}: {u.render()}" for u in recent)

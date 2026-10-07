"""What happened to every trade offer in a match.

Reads a run file and rebuilds each offer with everyone's answer to it, then
compares consecutive offers by the same proposer in the same turn. The question
it exists for: when an offer is turned down, does the proposer improve it, and
does anyone ever change their answer?

The one distinction everything rests on is *why* a seat said no. Only a
`decision` event is a model choosing. A `reflex` reject at a trade prompt means
REJECT was the only legal move — the seat did not hold what was asked for — and
the bot is deterministic. In the first real match 147 of 280 rejections were
forced and 99 were the bot, so a count that mixes them measures who happened to
hold brick, not who refused. The agents themselves cannot see this distinction:
at a real table a refusal does not come with its reason, and the prompt keeps
it that way.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")

#: How a response came about, keyed by event kind.
HOW = {
    "decision": "chose",
    "reflex": "forced",
    "bot": "bot",
    "own_offer_skipped": "own",
    "not_addressed": "not addressed",
    "fallback": "fallback",
}


@dataclass
class Offer:
    turn: int
    proposer: str
    persona: str
    give: Tuple[int, ...]
    want: Tuple[int, ...]
    say: str = ""
    #: color -> (answer, how); answer is "accept", "reject" or "counter"
    answers: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    #: color -> (give, want) from the countering seat's side
    counters: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = field(default_factory=dict)
    #: the one player this offer was put to: a counter-offer taken up
    to: Optional[str] = None
    #: "confirmed", "cancelled", or None when the offer died with nobody accepting
    closed: Optional[str] = None
    partner: Optional[str] = None

    @property
    def terms(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return self.give, self.want

    def accepted_by(self, how: Optional[str] = None) -> List[str]:
        return [c for c, (a, h) in self.answers.items()
                if a == "accept" and (how is None or h == how)]

    def rejected_by(self, how: str) -> List[str]:
        return [c for c, (a, h) in self.answers.items() if a == "reject" and h == how]

    @property
    def answerable(self) -> bool:
        """Whether any seat with a choice was asked. A seat that could not pay is
        forced to reject, and the bot's answer does not move with persuasion."""
        return any(h in ("chose", "fallback") for _, h in self.answers.values())

    def describe(self) -> str:
        return f"{_cards(self.give)} for {_cards(self.want)}"


def _cards(counts) -> str:
    parts = [f"{n} {r.lower()}" for r, n in zip(RESOURCES, counts) if n]
    return ", ".join(parts) or "nothing"


def offers(events: Iterable[dict]) -> List[Offer]:
    """Every offer in the order it was made, with its answers and outcome.

    The turn is read off the last `state` event rather than counted, because the
    runner emits one after every ply and it carries the engine's own number.
    """
    personas: Dict[str, str] = {}
    turn, out, current = None, [], None
    for e in events:
        kind = e["kind"]
        if kind == "game_start":
            personas = {s["color"]: s["name"] for s in e["seats"]}
            continue
        if kind == "state":
            turn = e["turn"]
            continue
        action = e.get("action") or {}
        kind_of_move = action.get("type")
        if kind_of_move == "OFFER_TRADE":
            value = action["value"]
            current = Offer(
                turn=turn, proposer=e["color"],
                persona=personas.get(e["color"], e.get("persona", "?")),
                give=tuple(value[:5]), want=tuple(value[5:10]),
                say=e.get("say") or "", to=e.get("to"),
            )
            out.append(current)
        elif current is None:
            continue
        elif kind_of_move in ("ACCEPT_TRADE", "REJECT_TRADE"):
            answer = "accept" if kind_of_move == "ACCEPT_TRADE" else "reject"
            if e.get("counter"):
                # To the engine a counter is a rejection; at the table it is not.
                answer = "counter"
                current.counters[e["color"]] = tuple(tuple(x) for x in e["counter"])
            current.answers[e["color"]] = (answer, HOW.get(kind, kind))
        elif kind_of_move == "CONFIRM_TRADE":
            current.closed = "confirmed"
            # value is the offer followed by the accepting colour, e.g. "Color.BLUE"
            current.partner = str(action["value"][-1]).rsplit(".", 1)[-1]
        elif kind_of_move == "CANCEL_TRADE":
            current.closed = "cancelled"
    return out


def change(before: Offer, after: Offer) -> str:
    """How `after` differs from `before`, seen from the side of whoever answers.

    'sweetened' gives more or asks less of the same resources; 'hardened' the
    reverse; 'reshaped' swaps what is traded. This is the proposer revising its
    own terms; a responder's counter-offer is recorded on the offer it answers.
    """
    if after.terms == before.terms:
        return "repeated"
    same_kinds = all(
        (a > 0) == (b > 0)
        for a, b in zip(before.give + before.want, after.give + after.want)
    )
    if same_kinds:
        gives_more = all(b >= a for a, b in zip(before.give, after.give))
        asks_less = all(b <= a for a, b in zip(before.want, after.want))
        if gives_more and asks_less:
            return "sweetened"
        gives_less = all(b <= a for a, b in zip(before.give, after.give))
        asks_more = all(b >= a for a, b in zip(before.want, after.want))
        if gives_less and asks_more:
            return "hardened"
    return "reshaped"


@dataclass
class Step:
    """One offer followed by the same proposer's next offer in the same turn."""
    before: Offer
    after: Offer
    how: str


def steps(all_offers: List[Offer]) -> List[Step]:
    by_turn = defaultdict(list)
    for o in all_offers:
        by_turn[(o.turn, o.proposer)].append(o)
    out = []
    for seq in by_turn.values():
        out.extend(Step(a, b, change(a, b)) for a, b in zip(seq, seq[1:]))
    return out


def mind_changes(all_offers: List[Offer]) -> List[Tuple[Offer, Offer, str]]:
    """(rejected, accepted, color): a seat that chose to reject an offer and then
    accepted the identical offer later in the same turn.

    Same turn only, on purpose: across turns the responder's hand has changed and
    accepting later says more about the dice than about the pitch.
    """
    by_turn = defaultdict(list)
    for o in all_offers:
        by_turn[(o.turn, o.proposer)].append(o)
    out = []
    for seq in by_turn.values():
        for i, first in enumerate(seq):
            for later in seq[i + 1:]:
                if later.terms != first.terms:
                    continue
                for color in first.rejected_by("chose"):
                    if later.answers.get(color, ("", ""))[0] == "accept":
                        out.append((first, later, color))
    return out


def summarize(all_offers: List[Offer]) -> Dict[str, dict]:
    """Per proposing persona, the counts the report prints."""
    rows: Dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for o in all_offers:
        r = rows[o.persona]
        r["offers"] += 1
        r["confirmed"] += o.closed == "confirmed"
        r["withdrawn_after_accept"] += o.closed == "cancelled" and bool(o.accepted_by())
        r["unanswerable"] += not o.answerable
        r["counters_received"] += len(o.counters)
        r["counter_taken_up"] += o.to is not None
        r["counter_taken_up_confirmed"] += o.to is not None and o.closed == "confirmed"
    for s in steps(all_offers):
        r = rows[s.before.persona]
        r[s.how] += 1
        if s.how == "repeated" and not s.before.answerable:
            r["repeated_unanswerable"] += 1
        if s.how != "repeated" and s.before.rejected_by("chose"):
            r["revised_after_a_chosen_no"] += 1
            r["revision_accepted"] += bool(s.after.accepted_by("chose"))
    return {k: dict(v) for k, v in rows.items()}

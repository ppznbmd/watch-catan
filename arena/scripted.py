"""A scripted agent that actually plays — and actually haggles.

It sees exactly what the model would see: the rendered prompt, nothing else. So
it has to read its own hand and its legal moves back out of that text, which is
a second, useful proof that the render is legible enough to decide from.

This is a stand-in for watching and for tests, not a Catan strategy. It spends
nothing.
"""

import random
import re
from typing import List, Optional

from arena.deciders import Decision

RESOURCE_WORDS = ("wood", "brick", "sheep", "wheat", "ore")


# --------------------------------------------------------------- reading the prompt


def legal_moves(user: str) -> List[tuple]:
    """[(index, description), ...] straight out of the rendered list."""
    moves = []
    for line in user.splitlines():
        m = re.match(r"\s*\[(\d+)\]\s+(.*)", line)
        if m:
            moves.append((int(m.group(1)), m.group(2)))
    return moves


def find(user: str, phrase: str) -> Optional[int]:
    for i, desc in legal_moves(user):
        if phrase in desc:
            return i
    return None


def my_hand(user: str) -> dict:
    """The 'hand: 2 wood, 1 ore' line belonging to the seat marked (you)."""
    lines = user.splitlines()
    for i, line in enumerate(lines):
        if "(you)" in line and i + 1 < len(lines):
            hand = {}
            m = re.search(r"hand: (.*?)(?: \||$)", lines[i + 1])
            if not m or m.group(1).strip() == "nothing":
                return hand
            for part in m.group(1).split(","):
                bits = part.strip().split()
                if len(bits) == 2 and bits[1] in RESOURCE_WORDS:
                    hand[bits[1]] = int(bits[0])
            return hand
    return {}


def offer_on_table(user: str) -> Optional[tuple]:
    """(who, what they give me, what they want) — parsed from the ON THE TABLE line."""
    m = re.search(
        r"ON THE TABLE: (\w+) is offering you (.*?) in exchange for (.*?)\.", user
    )
    if not m:
        return None

    def parse(text):
        out = {}
        for part in text.split(","):
            bits = part.strip().split()
            if len(bits) == 2 and bits[1] in RESOURCE_WORDS:
                out[bits[1]] = int(bits[0])
        return out

    return m.group(1), parse(m.group(2)), parse(m.group(3))


# --------------------------------------------------------------------- the agent


def heuristic_script(style: str = "trader", rng=None):
    """style: 'trader' haggles freely, 'tough' rejects the first offer it sees
    from each player, 'quiet' never opens a negotiation."""
    rng = rng or random.Random(0)
    squeezed = set()

    def script(i, system, user):
        moves = legal_moves(user)

        # 1. close a trade rather than let it lapse — index 0 here is CANCEL
        idx = find(user, "close the trade with")
        if idx is not None:
            return Decision(reasoning="somebody took it", choice=idx,
                            offer_give=[], offer_want=[], say="done — pleasure.")

        # 2. respond to an offer aimed at us
        table = offer_on_table(user)
        if table:
            who, gives, wants = table
            accept = find(user, "accept the offer")
            hand = my_hand(user)
            surplus = {r: n for r, n in hand.items() if n >= 2}
            worth_it = accept is not None and any(r in surplus for r in wants)
            if style == "tough" and who not in squeezed:
                squeezed.add(who)
                if "counter with your own terms" in user:
                    # what was asked of me, for twice what was offered
                    give = [r for r, n in wants.items() for _ in range(n)]
                    want = [r for r, n in gives.items() for _ in range(2 * n)]
                    return Decision(
                        reasoning=f"first price from {who} is never the price",
                        choice=-1, offer_give=give, offer_want=want,
                        say=f"double it, {who.lower()}, and we have a deal.")
                return Decision(
                    reasoning=f"first price from {who} is never the price",
                    choice=find(user, "reject the offer"),
                    offer_give=[], offer_want=[],
                    say=f"not at that rate, {who.lower()}. come back with two.")
            if worth_it:
                return Decision(
                    reasoning=f"I am long on {'/'.join(wants)} and short on {'/'.join(gives)}",
                    choice=accept, offer_give=[], offer_want=[],
                    say="that one I can do.")
            return Decision(
                reasoning="I do not have it spare, or I do not need what is offered",
                choice=find(user, "reject the offer"), offer_give=[], offer_want=[],
                say=f"I am short on {', '.join(wants) or 'that'} myself. pass.")

        # 3. our turn: a counter-offer somebody made us, if we can meet it
        idx = find(user, "counter-offer")
        if idx is not None and style != "quiet":
            return Decision(reasoning="their terms beat haggling on", choice=idx,
                            offer_give=[], offer_want=[], say="fine — at your price.")

        # 4. build if we can, otherwise try to trade for what we lack
        for phrase, why in (
            ("upgrade node", "a city is two points of board presence"),
            ("build a settlement", "more board, more dice"),
            ("buy a development", "nothing better to do with the cards"),
        ):
            idx = find(user, phrase)
            if idx is not None:
                return Decision(reasoning=why, choice=idx, offer_give=[],
                                offer_want=[], say="")

        if "write your own trade offer" in user and style != "quiet":
            hand = my_hand(user)
            surplus = sorted((n, r) for r, n in hand.items() if n >= 2)
            missing = [r for r in RESOURCE_WORDS if hand.get(r, 0) == 0]
            if surplus and missing:
                give = surplus[-1][1]
                want = rng.choice(missing)
                return Decision(
                    reasoning=f"sitting on {surplus[-1][0]} {give}, cannot build without {want}",
                    choice=-1, offer_give=[give], offer_want=[want],
                    say=f"one {give} for one {want} — I have {give} to spare and you know it.")

        idx = find(user, "build a road")
        if idx is not None and rng.random() < 0.4:
            return Decision(reasoning="stretch toward the next spot", choice=idx,
                            offer_give=[], offer_want=[], say="")

        for phrase in ("end your turn", "roll the dice"):
            idx = find(user, phrase)
            if idx is not None:
                return Decision(reasoning="nothing worth doing", choice=idx,
                                offer_give=[], offer_want=[], say="")

        # 5. anything else (robber, discards, initial placement): take a move
        choice = rng.choice(moves)[0] if moves else 0
        return Decision(reasoning="no strong preference", choice=choice,
                        offer_give=[], offer_want=[], say="")

    return script

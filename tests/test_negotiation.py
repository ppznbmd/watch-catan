"""The negotiation loop, end to end, without spending a token.

A counter-offer does not exist in the engine — a responder may only accept or
reject. These tests pin down the emulation: reject with words, the proposer
hears them and re-offers, and the per-turn budget stops the haggling.
"""

from catanatron.models.enums import ActionType
from catanatron.models.player import Color
from catanatron.players.value import ValueFunctionPlayer

from arena.deciders import ScriptedDecider
from arena.events import read_run
from arena.personas import COOPERATOR, HARD_BARGAINER
from arena.runner import bot_seat, llm_seat, run_match
from tests.helpers import can_offer, d, index_of, offer_on_table


def make_proposer():
    """Offers 1 wood for 1 ore whenever it can, then ends its turn."""

    def script(i, system, user):
        # DECIDE_ACCEPTEES first: index 0 there is CANCEL_TRADE, so falling
        # through to a default index would silently throw the trade away
        try:
            return d(index_of(user, "close the trade with"), say="deal")
        except AssertionError:
            pass
        if can_offer(user):
            return d(-1, say="wood for ore, straight across", give=["wood"], want=["ore"])
        if offer_on_table(user):
            return d(index_of(user, "reject the offer"), say="no thanks")
        for phrase in ("end your turn", "roll the dice"):
            try:
                return d(index_of(user, phrase))
            except AssertionError:
                continue
        return d(0)

    return ScriptedDecider(script)


def make_responder(accept_from_call: int):
    """Rejects with a reason, then accepts once it has said its piece."""
    state = {"rejections": 0}

    def script(i, system, user):
        if offer_on_table(user):
            try:
                accept = index_of(user, "accept the offer")
            except AssertionError:
                accept = None  # cannot afford it; rejecting is the only legal move
            if accept is not None and state["rejections"] >= accept_from_call:
                return d(accept, say="fine — but you owe me")
            state["rejections"] += 1
            return d(index_of(user, "reject the offer"), say="not ore. I'd do it for wheat")
        try:
            return d(index_of(user, "close the trade with"), say="done")
        except AssertionError:
            pass
        for phrase in ("end your turn", "roll the dice"):
            try:
                return d(index_of(user, phrase))
            except AssertionError:
                continue
        return d(0)

    return ScriptedDecider(script)


def test_offer_reply_reoffer_and_the_turn_cap(tmp_path):
    summary = run_match(
        seats=[
            (Color.RED, llm_seat(HARD_BARGAINER, make_proposer())),
            (Color.BLUE, llm_seat(COOPERATOR, make_responder(accept_from_call=1))),
            (Color.ORANGE, bot_seat(ValueFunctionPlayer)),
        ],
        runs_dir=tmp_path,
        seed=42,
        max_offers_per_turn=3,
    )
    events = read_run(summary["path"])

    offers = [e for e in events if e.get("action", {}).get("type") == "OFFER_TRADE"]
    assert offers, "the scripted proposer never got an offer onto the table"

    # the pitch travelled with the offer
    assert any("wood for ore" in e.get("say", "") for e in offers)

    # a rejection carried words, not just a verdict
    rejects = [e for e in events if e.get("action", {}).get("type") == "REJECT_TRADE"]
    assert any("I'd do it for wheat" in e.get("say", "") for e in rejects)

    # the proposer re-offered inside the same turn — that is the counter-offer
    by_turn = {}
    turn = 0
    for e in events:
        if e["kind"] == "state":
            turn = e["turn"]
        if e.get("action", {}).get("type") == "OFFER_TRADE":
            by_turn[turn] = by_turn.get(turn, 0) + 1
    assert max(by_turn.values()) > 1, "nobody ever re-offered within one turn"

    # and the budget held: never more than the cap in any single turn
    assert max(by_turn.values()) <= 3, f"offer cap breached: {by_turn}"


def test_a_trade_actually_closes(tmp_path):
    summary = run_match(
        seats=[
            (Color.RED, llm_seat(HARD_BARGAINER, make_proposer())),
            (Color.BLUE, llm_seat(COOPERATOR, make_responder(accept_from_call=0))),
            (Color.ORANGE, bot_seat(ValueFunctionPlayer)),
        ],
        runs_dir=tmp_path,
        seed=42,
        max_offers_per_turn=3,
    )
    events = read_run(summary["path"])
    confirms = [e for e in events if e.get("action", {}).get("type") == "CONFIRM_TRADE"]
    accepts = [e for e in events if e.get("action", {}).get("type") == "ACCEPT_TRADE"]
    assert accepts, "no offer was ever accepted"
    assert confirms, "an accepted offer was never closed"

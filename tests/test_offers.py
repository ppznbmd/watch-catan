"""The offer analysis: a refusal only counts as one if somebody chose it.

Most of these build a turn by hand, because the cases that matter — a forced
reject next to a chosen one, a revision after a no — are rare enough in a
scripted match that a test relying on one would pass without exercising them.
"""

import random

from catanatron.models.player import Color
from catanatron.players.value import ValueFunctionPlayer

from arena.deciders import ScriptedDecider
from arena.events import read_run
from arena.offers import change, mind_changes, offers, summarize
from arena.personas import ROSTER
from arena.runner import bot_seat, llm_seat, run_match
from arena.scripted import heuristic_script

SEATS = [
    {"color": "RED", "name": "Hard bargainer"},
    {"color": "BLUE", "name": "Cooperator"},
    {"color": "ORANGE", "name": "Quiet builder"},
    {"color": "WHITE", "name": "ValueFunctionPlayer"},
]


def vec(give=(), want=()):
    """A 10-slot offer from resource names, e.g. vec(["SHEEP"], ["BRICK"])."""
    order = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
    v = [0] * 10
    for r in give:
        v[order.index(r)] += 1
    for r in want:
        v[5 + order.index(r)] += 1
    return v


class Turn:
    """Writes events in the shape the runner and LLMPlayer emit them."""

    def __init__(self):
        self.events = [{"kind": "game_start", "seats": SEATS}]
        self.turn = None

    def at(self, turn):
        self.turn = turn
        self.events.append({"kind": "state", "turn": turn})
        return self

    def _act(self, kind, color, type_, value, say=""):
        self.events.append({"kind": kind, "color": color, "say": say,
                            "action": {"type": type_, "value": value}})
        self.events.append({"kind": "state", "turn": self.turn})

    def offer(self, color, v, say=""):
        self._act("decision", color, "OFFER_TRADE", v, say)
        return self

    def answer(self, color, accept, how="decision"):
        self._act(how, color, "ACCEPT_TRADE" if accept else "REJECT_TRADE", [])
        return self

    def confirm(self, proposer, v, partner):
        self._act("decision", proposer, "CONFIRM_TRADE", v + [f"Color.{partner}"])
        return self

    def cancel(self, proposer):
        self._act("decision", proposer, "CANCEL_TRADE", None)
        return self


def test_a_forced_rejection_is_not_counted_as_a_refusal():
    """A seat that cannot pay has REJECT as its only legal move and never reaches
    the model. Counting it as a refusal would make "repeated an offer nobody would
    take" and "repeated an offer nobody could take" the same number, and in the
    first real match most repeats were the second kind."""
    t = (Turn().at(6)
         .offer("BLUE", vec(["SHEEP"], ["WOOD"]))
         .answer("ORANGE", False, how="reflex")
         .answer("WHITE", False, how="bot")
         .answer("RED", False, how="reflex")
         .offer("BLUE", vec(["SHEEP"], ["WOOD"])))
    first, _ = offers(t.events)
    assert not first.answerable
    assert first.rejected_by("chose") == []
    row = summarize(offers(t.events))["Cooperator"]
    assert row["repeated"] == 1
    assert row["repeated_unanswerable"] == 1


def test_a_revision_is_judged_from_the_side_of_whoever_answers():
    """Giving more or asking less is a concession, the opposite is a squeeze.
    Mixing them up would report a hard bargainer who lowers their price after
    being accepted as someone who gave ground."""
    base = offers(Turn().at(1).offer("RED", vec(["WHEAT"], ["ORE"])).events)[0]

    def after(give, want):
        return offers(Turn().at(1).offer("RED", vec(give, want)).events)[0]

    assert change(base, after(["WHEAT"], ["ORE"])) == "repeated"
    assert change(base, after(["WHEAT", "WHEAT"], ["ORE"])) == "sweetened"
    assert change(after(["WHEAT", "WHEAT"], ["ORE"]), base) == "hardened"
    assert change(base, after(["SHEEP"], ["ORE"])) == "reshaped"
    assert change(base, after(["WHEAT", "SHEEP"], ["ORE"])) == "reshaped"


def test_only_a_chosen_no_followed_by_a_yes_to_the_same_terms_is_a_change_of_mind():
    """The metric is whether persistence persuades. A forced reject that turns
    into an accept is a hand that changed, and a yes to *better* terms is a
    response to a concession — neither is someone changing their mind."""
    t = (Turn().at(10)
         .offer("BLUE", vec(["SHEEP"], ["BRICK"]))
         .answer("ORANGE", False).answer("RED", False, how="reflex")
         .offer("BLUE", vec(["SHEEP"], ["BRICK"]))
         .answer("ORANGE", True).answer("RED", True)
         .confirm("BLUE", vec(["SHEEP"], ["BRICK"]), "ORANGE"))
    flips = mind_changes(offers(t.events))
    assert [c for _, _, c in flips] == ["ORANGE"]

    sweetened = (Turn().at(10)
                 .offer("BLUE", vec(["SHEEP"], ["BRICK"]))
                 .answer("ORANGE", False)
                 .offer("BLUE", vec(["SHEEP", "SHEEP"], ["BRICK"]))
                 .answer("ORANGE", True))
    assert mind_changes(offers(sweetened.events)) == []
    row = summarize(offers(sweetened.events))["Cooperator"]
    assert row["revised_after_a_chosen_no"] == 1
    assert row["revision_accepted"] == 1


def test_the_same_offer_in_a_later_turn_is_not_a_change_of_mind():
    """Between turns the dice have paid out, so a later yes says more about the
    responder's hand than about the pitch."""
    t = (Turn().at(10)
         .offer("BLUE", vec(["SHEEP"], ["BRICK"])).answer("ORANGE", False)
         .at(14)
         .offer("BLUE", vec(["SHEEP"], ["BRICK"])).answer("ORANGE", True))
    assert mind_changes(offers(t.events)) == []
    assert summarize(offers(t.events))["Cooperator"].get("repeated", 0) == 0


def test_withdrawing_an_offer_someone_accepted_is_counted():
    """Taking an offer back after a yes is the one move that reneges in the open;
    it is what the hard bargainer did to open a lower price in the first match."""
    t = (Turn().at(61)
         .offer("RED", vec(["WHEAT", "WHEAT"], ["ORE"])).answer("BLUE", True).cancel("RED")
         .offer("RED", vec(["WHEAT"], ["ORE"])).answer("BLUE", True)
         .confirm("RED", vec(["WHEAT"], ["ORE"]), "BLUE"))
    first, second = offers(t.events)
    assert first.closed == "cancelled"
    assert second.closed == "confirmed" and second.partner == "BLUE"
    row = summarize([first, second])["Hard bargainer"]
    assert row["withdrawn_after_accept"] == 1
    assert row["hardened"] == 1


def test_every_offer_in_a_played_match_is_found_with_an_answer_from_every_seat(tmp_path):
    """Run against a real run file, not a hand-built one: if the event shapes the
    runner writes drift from what this reads, offers go missing silently and the
    report still prints plausible numbers."""
    summary = run_match(
        seats=[
            (Color.RED, llm_seat(ROSTER["Hard bargainer"], ScriptedDecider(
                heuristic_script("tough", rng=random.Random(1)), label="dry-run"))),
            (Color.BLUE, llm_seat(ROSTER["Cooperator"], ScriptedDecider(
                heuristic_script("trader", rng=random.Random(2)), label="dry-run"))),
            (Color.ORANGE, bot_seat(ValueFunctionPlayer)),
        ],
        runs_dir=tmp_path, seed=7, max_offers_per_turn=3,
    )
    found = offers(read_run(summary["path"]))
    assert summary["offers"] > 0
    assert len(found) == summary["offers"]
    colors = set(summary["agents"])
    for o in found:
        assert set(o.answers) | {o.proposer} == colors

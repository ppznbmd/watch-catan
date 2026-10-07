"""Counter-offers: a seat answering an offer may name other terms, and the
proposer may put those terms back to that seat alone.

The engine has no counter-offer. To it a counter is a rejection and a counter
taken up is an ordinary offer, which the engine still puts to every seat. The
bookkeeping around that is where it can go wrong silently: a counter shown to
the wrong proposer, taken up twice, or a directed offer quietly accepted by a
third player would all still produce a plausible match.
"""

import json
import random
import re

import pytest
from catanatron.models.player import Color

from arena.bot import TradingValuePlayer
from arena.deciders import ScriptedDecider
from arena.offers import offers
from arena.personas import ROSTER
from arena.prompt import PROMPT_VERSION
from arena.runner import bot_seat, llm_seat, run_match
from arena.scripted import heuristic_script


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    sent = []

    def recording(style, rng_seed):
        script = heuristic_script(style, rng=random.Random(rng_seed))

        def wrapped(i, system, user):
            if "THAT DID NOT WORK" not in user:
                sent.append(user)
            return script(i, system, user)
        return ScriptedDecider(wrapped, label="dry-run")

    runs = tmp_path_factory.mktemp("runs")
    events = []
    for seed in range(6):
        summary = run_match(
            seats=[
                (Color.RED, llm_seat(ROSTER["Hard bargainer"], recording("tough", seed))),
                (Color.BLUE, llm_seat(ROSTER["Cooperator"], recording("trader", seed + 50))),
                (Color.ORANGE, llm_seat(ROSTER["Quiet builder"], recording("quiet", seed + 90))),
                (Color.WHITE, bot_seat(TradingValuePlayer)),
            ],
            runs_dir=runs, seed=seed,
        )
        events.append([json.loads(line) for line in open(summary["path"])])
    return events, sent


def test_a_match_records_the_prompt_version_and_the_offer_budget(played):
    """Without the version a run cannot say which game its agents were playing,
    and every comparison across a prompt change has to guess from the date."""
    for events in played[0]:
        start = events[0]
        assert start["prompt_version"] == PROMPT_VERSION
        assert start["max_offers_per_turn"] == 2


def test_a_counter_is_a_rejection_to_the_engine_and_a_counter_at_the_table(played):
    """Counted as a plain rejection, a counter-offer would vanish from every
    analysis of haggling — the behaviour it was added to observe."""
    events_by_match, _ = played
    counters = [e for events in events_by_match for e in events
                if e["kind"] == "decision" and e.get("counter")]
    # 5-6 across hash seeds (DESIGN.md, *Reproducibility*); 3-5 taken up.
    assert len(counters) >= 4, "the fixture never countered, so this proves nothing"
    for e in counters:
        assert e["action"]["type"] == "REJECT_TRADE"
    found = [o for events in events_by_match for o in offers(events) if o.counters]
    assert found and all(o.answers[c][0] == "counter" for o in found for c in o.counters)


def test_a_taken_counter_is_put_to_its_author_alone_on_its_own_terms(played):
    """The proposer takes up exactly what was countered, and only its author may
    accept. A third seat accepting would close a trade nobody negotiated."""
    events_by_match, _ = played
    taken = 0
    for events in events_by_match:
        for o in offers(events):
            if o.to is None:
                continue
            taken += 1
            countered = [p for p in offers(events)
                         if p.turn == o.turn and p.proposer == o.proposer and o.to in p.counters]
            assert countered, f"offer to {o.to} alone with no counter from {o.to} before it"
            give, want = countered[-1].counters[o.to]
            assert (o.give, o.want) == (tuple(want), tuple(give)), "terms not mirrored"
            for color, (answer, how) in o.answers.items():
                if color != o.to:
                    assert answer == "reject" and how in ("not addressed", "bot", "own"), \
                        f"{color} answered an offer put to {o.to} alone: {answer} ({how})"
            if o.closed == "confirmed":
                assert o.partner == o.to
    assert taken >= 2


def test_a_counter_is_taken_up_at_most_once(played):
    """Taking a counter up puts its terms to the table once. A counter left on
    offer after use would let a proposer spend its budget replaying one deal."""
    events_by_match, _ = played
    for events in events_by_match:
        made, taken = {}, {}
        for o in offers(events):
            for author, (give, want) in o.counters.items():
                key = (o.turn, o.proposer, author, tuple(want), tuple(give))
                made[key] = made.get(key, 0) + 1
            if o.to:
                key = (o.turn, o.proposer, o.to, o.give, o.want)
                taken[key] = taken.get(key, 0) + 1
        for key, n in taken.items():
            assert n <= made.get(key, 0), f"counter {key} taken up {n} times"


def test_a_proposer_is_only_offered_counters_made_to_it(played):
    """Checked on the prompts themselves: an option is only real if the table
    talk the same seat reads shows the counter it comes from."""
    _, sent = played
    for user in sent:
        me = re.match(r"TURN \d+ — you are (\w+)\.", user).group(1)
        options = re.findall(r"\[\d+\] take up (\w+)'s counter-offer", user)
        assert me not in options
        talk = user.split("TABLE TALK (recent):\n", 1)[1].split("\n\n", 1)[0]
        for who in options:
            assert re.search(rf"turn \d+: {who} counters {me}'s offer", talk), \
                f"{me} shown a counter from {who} that its table talk does not hold"
    per_prompt = [len(re.findall(r"take up \w+'s counter-offer", u)) for u in sent]
    assert sum(per_prompt) > 0


def test_a_seat_that_cannot_pay_cannot_counter(played):
    """A seat is only called about an offer when it could accept it; otherwise
    the reflex policy rejects for free. Counters therefore come only from seats
    that were asked, and the prompt must not invite one where it cannot follow."""
    _, sent = played
    for user in sent:
        if "You may instead counter" in user:
            assert "accept the offer on the table" in user

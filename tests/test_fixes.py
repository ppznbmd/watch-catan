"""The three defects found by running a real match, each pinned by a test."""

import random

import pytest
from catanatron.models.player import Color
from catanatron.players.value import ValueFunctionPlayer

from arena.colors import FULL_TABLE
from arena.deciders import DecisionFormatError, OpenAICompatDecider
from arena.events import read_run
from arena.personas import ROSTER
from arena.runner import bot_seat, llm_seat, run_match
from arena.scripted import heuristic_script
from tests.fake_provider import completion, fake_provider


# ------------------------------------------------- 1. a proposer never answers itself

def test_the_proposer_is_never_asked_about_its_own_offer(tmp_path):
    """Catanatron walks the table excluding whoever just answered, not whoever
    offered — so without the guard the proposer is polled on its own trade."""
    script = heuristic_script("trader", rng=random.Random(2))

    def responder(body, i):
        user = body["messages"][1]["content"]
        return completion(script(i, body["messages"][0]["content"], user).model_dump_json())

    with fake_provider(responder) as (url, _):
        decider = OpenAICompatDecider("fake", base_url=url, api_key="t", timeout=10,
                                      label="fake")
        summary = run_match(
            seats=[
                (Color.RED, llm_seat(ROSTER["Cooperator"], decider)),
                (Color.BLUE, llm_seat(ROSTER["Hard bargainer"], decider)),
                (Color.ORANGE, bot_seat(ValueFunctionPlayer)),
            ],
            runs_dir=tmp_path, seed=7, max_offers_per_turn=3,
        )
    events = read_run(summary["path"])

    # The automatic rejection is still a REJECT_TRADE in the log — that is the
    # guard working. What must never happen is a *model call* on your own offer.
    proposer = None
    for e in events:
        kind = (e.get("action") or {}).get("type")
        if kind == "OFFER_TRADE":
            proposer = e["color"]
        elif kind in ("ACCEPT_TRADE", "REJECT_TRADE") and e["kind"] == "decision":
            assert e["color"] != proposer, (
                f"{e['color']} spent a model call answering its own offer"
            )

    # the guard fired, and it was recorded rather than hidden
    skipped = [e for e in events if e["kind"] == "own_offer_skipped"]
    assert skipped, "the engine never asked a proposer about its own offer here"
    assert sum(a["self_offers_skipped"] for a in summary["agents"].values()) == len(skipped)
    # and it is always a rejection: accepting would close a trade with yourself
    assert all(e["action"]["type"] == "REJECT_TRADE" for e in skipped)


# --------------------------------------------------------- 2. bot seats are logged

def test_bot_seats_appear_in_the_log(tmp_path):
    from catanatron.models.player import RandomPlayer

    summary = run_match(
        seats=[(c, bot_seat(RandomPlayer)) for c in list(FULL_TABLE)[:3]],
        runs_dir=tmp_path, seed=11,
    )
    events = read_run(summary["path"])
    bot_events = [e for e in events if e["kind"] == "bot"]
    assert bot_events, "a table of bots produced no action events at all"
    assert {e["color"] for e in bot_events} == {c.value for c in list(FULL_TABLE)[:3]}
    assert all(e["action"]["described"] for e in bot_events)


def test_llm_seats_are_not_double_logged(tmp_path):
    """An LLM seat narrates itself with its reasoning; the observer must not
    also emit a bare action for it."""
    script = heuristic_script("quiet", rng=random.Random(4))

    def responder(body, i):
        return completion(
            script(i, body["messages"][0]["content"], body["messages"][1]["content"])
            .model_dump_json()
        )

    with fake_provider(responder) as (url, _):
        decider = OpenAICompatDecider("fake", base_url=url, api_key="t", timeout=10,
                                      label="fake")
        summary = run_match(
            seats=[
                (Color.RED, llm_seat(ROSTER["Quiet builder"], decider)),
                (Color.BLUE, bot_seat(ValueFunctionPlayer)),
            ],
            runs_dir=tmp_path, seed=3,
        )
    events = read_run(summary["path"])
    assert {e["color"] for e in events if e["kind"] == "bot"} == {"BLUE"}
    assert "RED" not in {e["color"] for e in events if e["kind"] == "bot"}


# ------------------------------------------------- 3. unusable replies are still billed

def test_a_malformed_reply_carries_the_usage_it_was_billed_for():
    with fake_provider(
        lambda body, i: completion("not json", prompt_tokens=900, completion_tokens=200)
    ) as (url, _):
        decider = OpenAICompatDecider("fake", base_url=url, api_key="t", timeout=5)
        with pytest.raises(DecisionFormatError) as caught:
            decider("S", "U")
    assert caught.value.usage["input_tokens"] == 900
    assert caught.value.usage["output_tokens"] == 200


def test_the_cost_report_counts_replies_it_could_not_use(tmp_path):
    """Half the replies are junk. The provider bills for all of them, so the
    summary must show all of them — otherwise the cheaper provider looks cheaper
    than it is, by exactly its failure rate."""
    script = heuristic_script("trader", rng=random.Random(1))
    seen = {"n": 0, "in": 0, "out": 0}

    def responder(body, i):
        seen["n"] += 1
        seen["in"] += 900
        seen["out"] += 200
        if seen["n"] % 2 == 0:
            return completion("not json", prompt_tokens=900, completion_tokens=200)
        d = script(i, body["messages"][0]["content"], body["messages"][1]["content"])
        return completion(d.model_dump_json(), prompt_tokens=900, completion_tokens=200)

    with fake_provider(responder) as (url, _):
        decider = OpenAICompatDecider("fake", base_url=url, api_key="t", timeout=10,
                                      label="fake")
        summary = run_match(
            seats=[
                (Color.RED, llm_seat(ROSTER["Cooperator"], decider)),
                (Color.BLUE, bot_seat(ValueFunctionPlayer)),
            ],
            runs_dir=tmp_path, seed=3,
        )

    usage = summary["agents"]["RED"]["usage"]
    assert usage["input_tokens"] == seen["in"], "billed input tokens went unreported"
    assert usage["output_tokens"] == seen["out"]
    assert summary["agents"]["RED"]["llm_calls"] == seen["n"]
    assert summary["agents"]["RED"]["malformed"] > 0


def test_the_match_summary_adds_up_the_cache_writes_it_was_billed_for(tmp_path):
    """The cost line prices writes at 1.25x from this sum. If one call's count
    were dropped, that call's input would be priced as if it had not been written."""
    script = heuristic_script("trader", rng=random.Random(1))
    seen = {"written": 0}

    def responder(body, i):
        seen["written"] += 880
        d = script(i, body["messages"][0]["content"], body["messages"][1]["content"])
        payload = completion(d.model_dump_json(), prompt_tokens=900, completion_tokens=200)
        payload["usage"]["prompt_tokens_details"] = {"cached_tokens": 0,
                                                     "cache_write_tokens": 880}
        return payload

    with fake_provider(responder) as (url, _):
        decider = OpenAICompatDecider("fake", base_url=url, api_key="t", timeout=10,
                                      label="fake")
        summary = run_match(
            seats=[
                (Color.RED, llm_seat(ROSTER["Cooperator"], decider)),
                (Color.BLUE, bot_seat(ValueFunctionPlayer)),
            ],
            runs_dir=tmp_path, seed=3,
        )

    assert summary["agents"]["RED"]["usage"]["cache_write_input_tokens"] == seen["written"]

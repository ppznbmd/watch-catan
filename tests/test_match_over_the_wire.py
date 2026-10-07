"""A whole match played through a real HTTP provider.

The fake server runs the heuristic agent and returns its answer as json, so the
entire stack is exercised for every decision: rendered prompt -> HTTP -> json ->
schema validation -> legal action -> engine. Every third reply is deliberately
unusable, in the shapes DeepSeek's docs warn about.
"""

import random

from catanatron.models.player import Color
from catanatron.players.value import ValueFunctionPlayer

from arena.deciders import OpenAICompatDecider
from arena.events import read_run
from arena.personas import ROSTER
from arena.runner import bot_seat, llm_seat, run_match
from arena.scripted import heuristic_script
from tests.fake_provider import completion, fake_provider

JUNK = ["", "sorry, I cannot help with that", '{"reasoning": "no choice key"}']


def responder_factory(junk_every=0, trace=None):
    script = heuristic_script("trader", rng=random.Random(1))
    seen = {"n": 0}

    def responder(body, i):
        seen["n"] += 1
        if junk_every and seen["n"] % junk_every == 0:
            return completion(JUNK[(seen["n"] // junk_every - 1) % len(JUNK)])
        user = body["messages"][1]["content"]
        decision = script(i, body["messages"][0]["content"], user)
        return completion(decision.model_dump_json(),
                          reasoning_content=trace and f"{trace} {seen['n']}")

    return responder


def play_over_the_wire(tmp_path, junk_every=0, seed=5, trace=None):
    with fake_provider(responder_factory(junk_every, trace)) as (url, state):
        decider = OpenAICompatDecider("fake-model", base_url=url, api_key="t",
                                      timeout=10.0, label="fake-model")
        summary = run_match(
            seats=[
                (Color.RED, llm_seat(ROSTER["Hard bargainer"], decider)),
                (Color.BLUE, llm_seat(ROSTER["Cooperator"], decider)),
                (Color.ORANGE, bot_seat(ValueFunctionPlayer)),
            ],
            runs_dir=tmp_path, seed=seed, max_offers_per_turn=3,
        )
    return summary, state


def test_a_clean_provider_plays_a_whole_match(tmp_path):
    summary, state = play_over_the_wire(tmp_path)
    events = read_run(summary["path"])

    assert events[-1]["kind"] == "game_over"
    assert summary["turns"] > 5
    assert state["calls"] > 20, "the provider was barely consulted"

    # every decision is attributed to the model that made it
    decisions = [e for e in events if e["kind"] == "decision"]
    assert decisions and all(e["model"] == "fake-model" for e in decisions)

    # negotiation survived the round trip, words and all
    offers = [e for e in events if e.get("action", {}).get("type") == "OFFER_TRADE"]
    assert offers, "no offer was authored over the wire"
    assert any(e.get("say") for e in offers), "the pitch did not survive the round trip"

    # token usage came back from the provider, so the cost line has real numbers
    assert summary["agents"]["RED"]["usage"]["input_tokens"] > 0
    assert summary["agents"]["RED"]["malformed"] == 0


def test_every_decision_logs_the_trace_it_came_with(tmp_path):
    """A seat whose provider thinks out loud leaves the trace in the event log,
    one per decision. It is the only record of what the model actually weighed:
    a match is not replayed to recover it."""
    summary, _ = play_over_the_wire(tmp_path, trace="thought")
    decisions = [e for e in read_run(summary["path"]) if e["kind"] == "decision"]
    assert decisions and all(e["thinking"].startswith("thought ") for e in decisions)
    # each decision carries its own trace, not a neighbour's
    assert len({e["thinking"] for e in decisions}) == len(decisions)


def test_a_provider_that_returns_junk_is_absorbed(tmp_path):
    summary, _ = play_over_the_wire(tmp_path, junk_every=3)
    events = read_run(summary["path"])

    malformed = [e for e in events if e["kind"] == "malformed"]
    assert malformed, "the junk replies were never caught"
    # all three documented failure shapes were seen and named
    complaints = " ".join(e["complaint"] for e in malformed)
    for expected in ("empty", "valid json", "wrong shape"):
        assert expected in complaints, f"never hit: {expected}"

    # and the match still finished
    assert events[-1]["kind"] == "game_over"
    assert summary["turns"] > 5
    assert summary["agents"]["RED"]["malformed"] > 0

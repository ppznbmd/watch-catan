"""The OpenAI-compatible path, over a real socket.

These cover the failures DeepSeek's own docs warn about: it guarantees valid
json, not the right shape, and it "may occasionally return empty content".
"""

import json

import pytest

from arena.deciders import DecisionFormatError, OpenAICompatDecider
from tests.fake_provider import completion, decision_json, fake_provider


def decider_against(url, **kwargs):
    return OpenAICompatDecider("fake-model", base_url=url, api_key="test",
                               timeout=5.0, **kwargs)


def test_a_well_formed_reply_round_trips():
    with fake_provider(lambda body, i: completion(decision_json(choice=2, say="deal"))) as (url, state):
        decision, usage = decider_against(url)("SYSTEM", "USER")
    assert decision.choice == 2
    assert decision.say == "deal"
    assert usage.pop("requested_at") > 0, "DeepSeek's rate depends on the hour"
    assert usage == {"input_tokens": 894, "cached_input_tokens": 0,
                     "output_tokens": 120, "model": "fake-model"}

    # the request we actually put on the wire
    body = state["requests"][0]
    assert body["model"] == "fake-model"
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["content"] == "USER"
    # DeepSeek requires the literal word "json" to appear in the prompt
    assert "json" in body["messages"][0]["content"].lower()


def test_cache_hit_tokens_are_read_back():
    """DeepSeek reports the cached share of the prompt, and it is priced ~50x
    lower — so the cost line is wrong if we do not read it."""
    payload = completion(decision_json())
    payload["usage"]["prompt_cache_hit_tokens"] = 410
    payload["usage"]["prompt_cache_miss_tokens"] = 484
    with fake_provider(lambda body, i: payload) as (url, _):
        _, usage = decider_against(url)("S", "U")
    assert usage["cached_input_tokens"] == 410
    assert usage["input_tokens"] == 894


def test_cache_writes_are_read_back_when_the_provider_reports_them():
    """GPT-5.6 bills a cache write at 1.25x the input rate and writes every
    uncached prompt by default. Dropping the count prices a match 12% low, as the
    v3 baseline was until the billing dashboard showed it."""
    payload = completion(decision_json(), prompt_tokens=2316)
    payload["usage"]["prompt_tokens_details"] = {"cached_tokens": 0, "cache_write_tokens": 2313}
    with fake_provider(lambda body, i: payload) as (url, _):
        _, usage = decider_against(url)("S", "U")
    assert usage["cache_write_input_tokens"] == 2313


def test_a_provider_that_does_not_report_cache_writes_leaves_the_count_out():
    """Absent is not zero. A 0 would tell the pricing that nothing was written,
    and price an unreported write at the plain input rate."""
    with fake_provider(lambda body, i: completion(decision_json())) as (url, _):
        _, usage = decider_against(url)("S", "U")
    assert "cache_write_input_tokens" not in usage


def test_the_reasoning_trace_is_kept_beside_the_decision():
    """Flash returns its thinking as `reasoning_content`, apart from the json.
    Dropped, the only reasoning left is the two-sentence summary written for
    the spectator, and a lie cannot be told from a mistake on that."""
    payload = completion(decision_json(reasoning="summary"), reasoning_content="the whole trace")
    with fake_provider(lambda body, i: payload) as (url, _):
        decision, _ = decider_against(url)("S", "U")
    assert decision.thinking == "the whole trace"
    assert decision.reasoning == "summary"


def test_a_reply_without_a_trace_has_no_thinking():
    """Absent is None, not "": a model that does not think out loud must not
    look like one that thought nothing."""
    with fake_provider(lambda body, i: completion(decision_json())) as (url, _):
        decision, _ = decider_against(url)("S", "U")
    assert decision.thinking is None


def test_a_decider_can_leave_the_reasoning_summary_out_of_the_contract():
    """The lie trap asks for no summary, so the trace is the only reasoning. If
    the contract still asked for one, the model would still be writing for a
    reader, and a reply without it would be thrown away as malformed."""
    reply = json.dumps({"choice": 1, "offer_give": [], "offer_want": [], "say": "hi"})
    with fake_provider(lambda body, i: completion(reply, reasoning_content="trace")) as (url, state):
        decision, _ = decider_against(url, ask_reasoning=False)("S", "U")
    assert decision.choice == 1 and decision.reasoning == ""
    assert decision.thinking == "trace"
    assert '"note"' not in state["requests"][0]["messages"][0]["content"]


def test_a_match_decider_still_refuses_a_reply_without_reasoning():
    """A seat's summary is shown back to it next turn as its own note; a reply
    that drops it must still be caught, not silently filled with ""."""
    reply = json.dumps({"choice": 1, "offer_give": [], "offer_want": [], "say": ""})
    with fake_provider(lambda body, i: completion(reply)) as (url, _):
        with pytest.raises(DecisionFormatError, match="wrong shape: note"):
            decider_against(url)("S", "U")


def test_the_model_is_asked_for_a_note_to_itself_and_never_told_of_a_reader():
    """Asking for `reasoning` told the model its thinking had a reader, and a
    model that suspects one starts guessing what the reader wants. If the word
    came back, matches would again be played for an audience."""
    with fake_provider(lambda body, i: completion(decision_json(reasoning="plan: city"))) as (url, state):
        decision, _ = decider_against(url)("S", "U")
    assert decision.reasoning == "plan: city", "the note is what the seat rereads"
    contract = state["requests"][0]["messages"][0]["content"]
    assert '"note"' in contract
    assert "reasoning" not in contract and "spectator" not in contract


def test_effort_is_only_sent_when_asked():
    with fake_provider(lambda body, i: completion(decision_json())) as (url, state):
        decider_against(url)("S", "U")
        decider_against(url, effort="low")("S", "U")
    assert "reasoning_effort" not in state["requests"][0]
    assert state["requests"][1]["reasoning_effort"] == "low"


@pytest.mark.parametrize(
    "content,finish,expected",
    [
        ("", "stop", "the reply was empty"),
        ("   ", "stop", "the reply was empty"),
        ("not json at all", "stop", "not valid json"),
        ('["a", "list"]', "stop", "not an object"),
        ('{"reasoning": "hm"}', "stop", "wrong shape"),
        ('{"reasoning":"x","choice":"two"}', "stop", "wrong shape"),
        ('{"reasoning": "x", "choice": 1', "length", "cut off"),
        # A thinking model can spend the whole budget reasoning and return
        # nothing. That is a budget problem, not DeepSeek's documented empty reply.
        ("", "length", "still reasoning"),
    ],
)
def test_bad_replies_raise_a_retryable_format_error(content, finish, expected):
    with fake_provider(lambda body, i: completion(content, finish_reason=finish)) as (url, _):
        with pytest.raises(DecisionFormatError) as caught:
            decider_against(url)("S", "U")
    assert expected in str(caught.value)


def test_missing_optional_fields_are_filled_in():
    """A model that omits offer_give/say is answering usably — do not punish it."""
    minimal = json.dumps({"reasoning": "fine", "choice": 0})
    with fake_provider(lambda body, i: completion(minimal)) as (url, _):
        decision, _ = decider_against(url)("S", "U")
    assert decision.offer_give == [] and decision.offer_want == [] and decision.say == ""


def test_an_http_error_is_not_a_format_error():
    """A 500 is a transport failure: it must not be retried as if the model
    had said something wrong."""
    with fake_provider(lambda body, i: (500, {"error": {"message": "boom"}})) as (url, _):
        with pytest.raises(Exception) as caught:
            decider_against(url)("S", "U")
    assert not isinstance(caught.value, DecisionFormatError)


# ---------------------------------------------------------- schema-enforced path

def schema_decider(url, **kwargs):
    """What a gpt-* seat builds: schema enforced, and the current token param."""
    return OpenAICompatDecider(
        "gpt-5.6-luna", base_url=url, api_key="test", timeout=5.0,
        schema_enforced=True, token_param="max_completion_tokens", **kwargs
    )


def test_schema_mode_sends_a_json_schema_and_the_current_token_param():
    with fake_provider(lambda body, i: completion(decision_json(choice=1, say="ok"))) as (url, state):
        decision, _ = schema_decider(url)("SYSTEM", "USER")
    assert decision.choice == 1

    body = state["requests"][0]
    assert body["response_format"]["type"] == "json_schema"
    assert "schema" in body["response_format"]["json_schema"]
    # max_tokens is deprecated and rejected by reasoning models
    assert "max_completion_tokens" in body and "max_tokens" not in body
    # the provider validates the shape, so the prompt does not repeat the contract
    assert "Reply with a single json object" not in body["messages"][0]["content"]
    # the trace is ours to keep, never a field the model is asked to fill
    assert "thinking" not in json.dumps(body["response_format"])


def test_a_refusal_is_a_format_error_not_a_crash():
    payload = completion(None)
    payload["choices"][0]["message"]["refusal"] = "I won't do that"
    with fake_provider(lambda body, i: payload) as (url, _):
        with pytest.raises(DecisionFormatError) as caught:
            schema_decider(url)("S", "U")
    assert "refused" in str(caught.value)


def test_truncation_is_caught_in_schema_mode_too():
    with fake_provider(lambda body, i: completion(decision_json(), finish_reason="length")) as (url, _):
        with pytest.raises(DecisionFormatError) as caught:
            schema_decider(url)("S", "U")
    assert "cut off" in str(caught.value)


def test_deepseek_still_gets_the_contract_and_the_old_token_param():
    with fake_provider(lambda body, i: completion(decision_json())) as (url, state):
        decider_against(url)("SYSTEM", "USER")
    body = state["requests"][0]
    assert body["response_format"] == {"type": "json_object"}
    assert "max_tokens" in body and "max_completion_tokens" not in body
    assert "Reply with a single json object" in body["messages"][0]["content"]


def test_a_deepseek_seat_has_room_to_reason_past_the_old_4096_cap(monkeypatch):
    """deepseek-flash reasoned past 4096 tokens on 29% of the road positions
    re-asked on 2026-09-18, and once reached 16123. At 4096 those decisions came
    back empty and fell to a retry or the reflex policy."""
    from arena.deciders import make_decider
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    decider = make_decider("deepseek-flash")
    assert decider._max_tokens >= 32768
    assert decider._client.timeout >= 300


def test_an_explicit_token_limit_still_wins_over_the_provider_default(monkeypatch):
    """scripts that ask for a specific budget must get it."""
    from arena.deciders import make_decider
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    assert make_decider("deepseek-flash", max_tokens=1000)._max_tokens == 1000


def test_a_claude_seat_keeps_its_own_budget(monkeypatch):
    """The provider budget belongs to the OpenAI-compatible path; passing it to
    the Anthropic client would crash every Claude seat at construction."""
    pytest.importorskip("anthropic")
    from arena.deciders import make_decider
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert make_decider("claude-haiku-4-5")._max_tokens == 8192


def test_provider_switches_reach_the_wire_beside_the_effort():
    """DeepSeek turns its thinking mode off through a body field the SDK has no
    argument for. If it were dropped, a no-thinking comparison would silently
    measure the thinking model again."""
    with fake_provider(lambda body, i: completion(decision_json())) as (url, state):
        decider_against(url, effort="low", extra_body={"thinking": {"type": "disabled"}})("S", "U")
    body = state["requests"][0]
    assert body["thinking"] == {"type": "disabled"}
    assert body["reasoning_effort"] == "low"


def test_a_match_decider_asks_gpt_5_6_not_to_write_the_cache(monkeypatch):
    """A match never sends the same prompt twice, so every write is paid for at
    1.25x and never read: ~11% of a match's bill."""
    from arena.deciders import make_decider

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    off = {"prompt_cache_options": {"mode": "explicit"}}
    assert make_decider("gpt-5.6-luna", cache_writes=False)._extra_body == off
    assert make_decider("gpt-5.6-luna")._extra_body is None, \
        "re-asking scripts would lose their cache hits"
    assert make_decider("deepseek-flash", cache_writes=False)._extra_body is None, \
        "DeepSeek has no write fee and no such option"
    budget = {"thinking": {"type": "disabled"}}
    assert make_decider("gpt-5.6-luna", cache_writes=False, extra_body=budget)._extra_body \
        == {**budget, **off}


def test_the_cache_option_reaches_the_wire_on_the_schema_path():
    """OpenAI requests go through `parse`, not `create`; the option was verified
    against the real API only through `create`."""
    off = {"prompt_cache_options": {"mode": "explicit"}}
    with fake_provider(lambda body, i: completion(decision_json())) as (url, state):
        decider_against(url, schema_enforced=True, extra_body=off)("S", "U")
    assert state["requests"][0]["prompt_cache_options"] == {"mode": "explicit"}


# ------------------------------------------------------------------------ flex

def flex_payload(tier="flex"):
    payload = completion(decision_json())
    payload["service_tier"] = tier
    return payload


def test_flex_is_asked_for_on_the_wire_and_the_served_tier_is_read_back():
    """The half price is billed on the tier OpenAI says it served. If the request
    field were dropped, every flex run would pay full price and be reported at
    half."""
    with fake_provider(lambda body, i: flex_payload()) as (url, state):
        _, usage = schema_decider(url, service_tier="flex")("S", "U")
    assert state["requests"][0]["service_tier"] == "flex"
    assert usage["service_tier"] == "flex"


def test_a_standard_request_does_not_ask_for_a_tier():
    with fake_provider(lambda body, i: completion(decision_json())) as (url, state):
        _, usage = schema_decider(url)("S", "U")
    assert "service_tier" not in state["requests"][0]
    assert "service_tier" not in usage


def test_a_flex_refusal_for_capacity_is_waited_out():
    """OpenAI answers flex with a 429 when it has no capacity, and does not bill
    it. Treated as a transport failure, one busy minute would hand a whole
    stretch of a batch to the fallback."""
    busy = (429, {"error": {"message": "Resource Unavailable", "type": "rate_limit"}})
    with fake_provider(lambda body, i: busy if i < 4 else flex_payload()) as (url, state):
        decision, usage = schema_decider(url, service_tier="flex", flex_waits=(0, 0))("S", "U")
    assert state["calls"] == 5
    assert usage["service_tier"] == "flex"


def test_flex_that_stays_busy_fails_as_transport_and_never_falls_back_to_full_price():
    """Re-sending at the standard tier would double the price without anyone
    having chosen to pay it."""
    busy = (429, {"error": {"message": "Resource Unavailable", "type": "rate_limit"}})
    with fake_provider(lambda body, i: busy) as (url, state):
        with pytest.raises(Exception) as caught:
            schema_decider(url, service_tier="flex", flex_waits=(0,))("S", "U")
    assert not isinstance(caught.value, DecisionFormatError)
    assert all(r.get("service_tier") == "flex" for r in state["requests"])


def test_flex_is_refused_for_a_model_without_it(monkeypatch):
    """DeepSeek has no flex tier. Ignoring the flag would run a batch at full
    price that its author believed was half."""
    from arena.deciders import make_decider

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    assert make_decider("gpt-5.6-sol", flex=True)._service_tier == "flex"
    assert make_decider("gpt-5.6-sol", flex=True)._client.timeout >= 900
    with pytest.raises(ValueError):
        make_decider("deepseek-flash", flex=True)

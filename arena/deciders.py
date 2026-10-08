"""The model call, behind one interface, so any provider can take a seat.

Two provider shapes are supported:

- OpenAI-compatible chat completions (DeepSeek, and anything else that speaks it)
- Anthropic's Messages API

They differ in one way that matters. Anthropic enforces the response schema
server-side; DeepSeek guarantees only that the reply is *valid JSON*, not that it
has the right shape, and its docs warn it "may occasionally return empty
content". So the validation here is not a safety net — for DeepSeek it is the
mechanism. A format failure is raised as `DecisionFormatError`, which the caller
retries by telling the model what was wrong; anything else is a transport
failure and goes straight to the fallback policy.
"""

import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError


class DecisionFormatError(Exception):
    """The reply came back, but not in a shape we can use. Worth one retry.

    Carries the usage of the call that produced it: the provider billed for this
    reply even though we cannot use it, and a cost report that drops it
    understates the bill — by exactly the failure rate, which is the number you
    are comparing providers on.
    """

    def __init__(self, message, usage=None):
        super().__init__(message)
        self.usage = usage or {"input_tokens": 0, "cached_input_tokens": 0,
                               "output_tokens": 0}


class Decision(BaseModel):
    """What an agent hands back for one decision."""

    # The model fills `note`, a line to itself that only it will reread; asking
    # for `reasoning` told it somebody reads its reasoning, and a model that
    # suspects an audience starts guessing what the audience wants (prompt v4).
    # Stored and logged as `reasoning`, the name everything downstream reads.
    model_config = ConfigDict(populate_by_name=True)

    reasoning: str = Field(
        alias="note",
        description="Two or three sentences to yourself, shown back to you on "
                    "your next turn and to no other player.")
    choice: int = Field(
        description="Index of the legal move you pick, or -1 to author your own trade offer."
    )
    offer_give: List[str] = Field(
        default_factory=list,
        description="Resource names you give, only when choice is -1. Otherwise empty.",
    )
    offer_want: List[str] = Field(
        default_factory=list,
        description="Resource names you want, only when choice is -1. Otherwise empty.",
    )
    say: str = Field(
        default="",
        description="What you say out loud, heard by everyone. Empty when not negotiating.",
    )
    # The provider's own reasoning trace, when it returns one (DeepSeek's
    # `reasoning_content`). Private, so it never enters the schema a provider
    # is asked to fill: `reasoning` is the summary the model writes knowing it
    # is read, and on the first Flash reply checked it was 2 sentences against
    # 10,961 characters of trace that said things the summary left out.
    _thinking: Optional[str] = PrivateAttr(default=None)

    @property
    def thinking(self) -> Optional[str]:
        return self._thinking


# Providers without schema enforcement need the contract spelled out, and
# DeepSeek additionally requires the literal word "json" to appear in the prompt.
JSON_CONTRACT = """
Reply with a single json object and nothing else — no prose around it, no code fence.
It must have exactly these keys:

{"note": "<two or three sentences to yourself>",
 "choice": <integer index of your move, or -1 to write your own trade offer>,
 "offer_give": ["<resource>", ...],
 "offer_want": ["<resource>", ...],
 "say": "<what you say out loud, or an empty string>"}

Use [] for offer_give and offer_want and "" for say when they do not apply.
"""

#: The same contract without `reasoning`, for a probe where the provider's own
#: trace is the only reasoning wanted: the summary is written for whoever reads
#: it, and asking for one tells the model somebody will.
JSON_CONTRACT_NO_REASONING = JSON_CONTRACT.replace(
    '{"note": "<two or three sentences to yourself>",\n "choice"', '{"choice"')
assert JSON_CONTRACT_NO_REASONING != JSON_CONTRACT


class Decider(Protocol):
    label: str

    def __call__(self, system: str, user: str) -> tuple[Decision, dict]: ...


# --------------------------------------------------------------------- providers


@dataclass(frozen=True)
class Provider:
    name: str
    api_key_env: str
    base_url: Optional[str]
    kind: str  # "openai" | "anthropic"
    schema_enforced: bool
    prefixes: tuple
    #: Chat Completions renamed this; reasoning models reject the old name.
    token_param: str = "max_tokens"
    #: The output budget covers the reasoning too, so a model that thinks at
    #: length needs room for it or it answers with nothing.
    max_tokens: int = 4096
    #: Seconds per request; must fit `max_tokens` at the provider's speed.
    timeout: float = 120.0
    #: Models whose cache writes can be turned off with an explicit cache mode
    #: and no breakpoint, and the request body that does it.
    cache_off_prefixes: tuple = ()
    cache_off_body: Optional[dict] = None
    #: Models with a flex tier: half price, slower, and sometimes refused.
    flex_prefixes: tuple = ()


PROVIDERS = (
    # OpenAI enforces a json schema, so a malformed reply is not a failure mode
    # we have to handle there at all. It also rejects `max_tokens` on reasoning
    # models — `max_completion_tokens` is the current name.
    Provider("openai", "OPENAI_API_KEY", os.environ.get("OPENAI_BASE_URL"),
             "openai", True, ("gpt", "o1", "o3", "o4"),
             token_param="max_completion_tokens",
             # GPT-5.6 writes every uncached prompt to its cache at 1.25x the
             # input rate. Sent this, it wrote 0 of 2,316 tokens (2026-09-18).
             cache_off_prefixes=("gpt-5.6",),
             cache_off_body={"prompt_cache_options": {"mode": "explicit"}},
             # The pricing page lists flex rates, half the standard ones, for
             # all three GPT-5.6 models (2026-09-24). DeepSeek has no such tier;
             # its discount is off-peak hours.
             flex_prefixes=("gpt-5.6",)),
    # DeepSeek guarantees valid json, not the right shape. deepseek-flash thinks
    # by default: on 120 re-asked road positions it passed 4096 output tokens 29%
    # of the time and reached 16123, at ~200 tokens/s. Its own ceiling is 384K.
    Provider("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com",
             "openai", False, ("deepseek",), max_tokens=32768, timeout=300.0),
    Provider("anthropic", "ANTHROPIC_API_KEY", None,
             "anthropic", True, ("claude",)),
)

BY_NAME = {p.name: p for p in PROVIDERS}


def resolve(spec: str) -> tuple[Provider, str]:
    """'deepseek-flash' -> (deepseek, 'deepseek-flash').
    'anthropic:claude-haiku-4-5' -> (anthropic, 'claude-haiku-4-5')."""
    if ":" in spec:
        name, _, model = spec.partition(":")
        if name not in BY_NAME:
            raise ValueError(
                f"unknown provider {name!r}; known: {', '.join(BY_NAME)}"
            )
        return BY_NAME[name], model
    for provider in PROVIDERS:
        if any(spec.startswith(p) for p in provider.prefixes):
            return provider, spec
    raise ValueError(
        f"cannot tell which provider serves {spec!r} — write it as 'provider:model' "
        f"({', '.join(BY_NAME)})"
    )


# ----------------------------------------------------------------------- clients


def trace(message) -> Optional[str]:
    """The reasoning trace a chat completion carries outside `content`, or None.
    The SDK does not model the field, so it arrives as an extra."""
    text = getattr(message, "reasoning_content", None) or (
        (getattr(message, "model_extra", None) or {}).get("reasoning_content"))
    return text or None


class OpenAICompatDecider:
    """Chat completions, JSON-object mode, validated here because the provider
    does not validate shape for us."""

    def __init__(self, model, base_url=None, api_key=None, effort=None,
                 max_tokens=4096, label=None, timeout=120.0,
                 schema_enforced=False, token_param="max_tokens", extra_body=None,
                 service_tier=None, flex_waits=(15, 30, 60, 120, 240),
                 ask_reasoning=True):
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key or "unused", base_url=base_url, timeout=timeout, max_retries=2
        )
        self._model = model
        self._effort = self.effort = effort
        self._max_tokens = max_tokens
        self._schema_enforced = schema_enforced
        self._token_param = token_param
        # Provider switches the OpenAI SDK has no argument for, such as
        # DeepSeek's {"thinking": {"type": "disabled"}}.
        self._extra_body = extra_body
        self._service_tier = self.service_tier = service_tier
        # Seconds to wait before each new try when flex has no capacity. OpenAI
        # answers that with a 429 it does not bill, and suggests backing off.
        self._flex_waits = flex_waits
        if schema_enforced and not ask_reasoning:
            raise ValueError("the enforced schema requires `reasoning`")
        self._contract = JSON_CONTRACT if ask_reasoning else JSON_CONTRACT_NO_REASONING
        self._ask_reasoning = ask_reasoning
        self.label = label or model

    def _send(self, method, **kwargs):
        """One request. On flex, a refusal for capacity is waited out and tried
        again; once the waits run out it is raised like any transport failure,
        never quietly re-sent at the standard price."""
        if self._service_tier != "flex":
            return method(**kwargs)
        from openai import RateLimitError

        for wait in (*self._flex_waits, None):
            try:
                return method(**kwargs)
            except RateLimitError:
                if wait is None:
                    raise
                time.sleep(wait)

    def __call__(self, system: str, user: str) -> tuple[Decision, dict]:
        self._requested_at = time.time()
        # With a schema the provider validates the shape for us, and spelling the
        # contract out again in the prompt only wastes tokens.
        system_content = system if self._schema_enforced else system + "\n" + self._contract
        kwargs = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user},
            ],
            self._token_param: self._max_tokens,
        }
        if self._effort:
            kwargs["reasoning_effort"] = self._effort
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        if self._service_tier:
            kwargs["service_tier"] = self._service_tier

        if self._schema_enforced:
            # The SDK raises its own errors for a truncated or filtered reply,
            # before we get to look at the response. Both are "the model answered
            # unusably", which is the retryable kind.
            from openai import ContentFilterFinishReasonError, LengthFinishReasonError

            try:
                response = self._send(self._client.chat.completions.parse,
                                      response_format=Decision, **kwargs)
            except LengthFinishReasonError as exc:
                raise DecisionFormatError(
                    "the reply was cut off before the json closed",
                    self._usage(getattr(exc, "completion", None)),
                ) from exc
            except ContentFilterFinishReasonError as exc:
                raise DecisionFormatError(
                    "the reply was blocked by a content filter",
                    self._usage(getattr(exc, "completion", None)),
                ) from exc
            billed = self._usage(response)
            if not response.choices:
                raise DecisionFormatError("the reply had no choices", billed)
            message = response.choices[0].message
            if getattr(message, "refusal", None):
                raise DecisionFormatError(f"the model refused: {message.refusal}", billed)
            if response.choices[0].finish_reason == "length":
                raise DecisionFormatError(
                    "the reply was cut off before the json closed", billed)
            if message.parsed is None:
                raise DecisionFormatError(
                    "the schema-validated reply came back empty", billed)
            message.parsed._thinking = trace(message)
            return message.parsed, {**billed, "model": self.label}

        kwargs["response_format"] = {"type": "json_object"}
        response = self._send(self._client.chat.completions.create, **kwargs)
        billed = self._usage(response)

        if not response.choices:
            raise DecisionFormatError("the reply had no choices", billed)
        message = response.choices[0].message
        content = (message.content or "").strip()
        # Length first: an empty reply that hit the limit spent the budget
        # reasoning, and reporting it as DeepSeek's documented empty reply hid
        # that for 33 answers in a row.
        if response.choices[0].finish_reason == "length":
            raise DecisionFormatError(
                "the reply was cut off before the json closed" if content else
                "the reply was cut off: the token limit ran out while the model "
                "was still reasoning", billed)
        if not content:
            # documented DeepSeek behaviour, not an exotic edge case
            raise DecisionFormatError("the reply was empty", billed)

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise DecisionFormatError(f"that was not valid json ({exc})", billed) from exc
        if not isinstance(payload, dict):
            raise DecisionFormatError("the json was not an object", billed)
        if not self._ask_reasoning:
            # Not asked for, so not missing. One written anyway is kept.
            payload.setdefault("reasoning", "")
        try:
            decision = Decision.model_validate(payload)
        except ValidationError as exc:
            missing = ", ".join(
                ".".join(str(p) for p in e["loc"]) for e in exc.errors()
            )
            raise DecisionFormatError(
                f"the json was the wrong shape: {missing}", billed) from exc

        decision._thinking = trace(message)
        return decision, {**billed, "model": self.label}

    def _usage(self, response) -> dict:
        usage = getattr(response, "usage", None) if response is not None else None
        extra = getattr(usage, "model_extra", None) or {}
        # DeepSeek caches prefixes automatically and reports the split; a cache
        # hit costs ~50x less than a miss, so the two are worth counting apart.
        cached = (
            getattr(usage, "prompt_cache_hit_tokens", None)
            or extra.get("prompt_cache_hit_tokens")
            or getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
            or 0
        )
        billed = {
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "cached_input_tokens": cached,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }
        # GPT-5.6 writes every uncached prompt to its cache by default and bills
        # the write at 1.25x the input rate; a match never reuses one. Kept apart
        # from a reported 0, because absent means "unknown", not "none".
        written = getattr(getattr(usage, "prompt_tokens_details", None),
                          "cache_write_tokens", None)
        if written is not None:
            billed["cache_write_input_tokens"] = written
        # The tier the provider says it served, which is what it bills. Asking
        # for flex is not proof of getting it, so only a reported tier is kept.
        tier = getattr(response, "service_tier", None) if response is not None else None
        if tier:
            billed["service_tier"] = tier
        # DeepSeek's price depends on the hour the request was made, and nothing
        # in its response says which rate applied.
        billed["requested_at"] = round(getattr(self, "_requested_at", time.time()), 3)
        return billed


class AnthropicDecider:
    """Claude behind `messages.parse`, where the schema is enforced server-side."""

    def __init__(self, model, effort=None, api_key=None, label=None, max_tokens=8192):
        import anthropic

        self._client = anthropic.Anthropic(**({"api_key": api_key} if api_key else {}))
        self._model = model
        self._effort = self.effort = effort
        self._max_tokens = max_tokens
        self.label = label or model

    def __call__(self, system: str, user: str) -> tuple[Decision, dict]:
        kwargs = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            thinking={"type": "adaptive"},
            output_format=Decision,
        )
        if self._effort:
            kwargs["output_config"] = {"effort": self._effort}
        response = self._client.messages.parse(**kwargs)
        billed = {
            "input_tokens": response.usage.input_tokens,
            "cached_input_tokens": getattr(
                response.usage, "cache_read_input_tokens", 0) or 0,
            "output_tokens": response.usage.output_tokens,
        }
        if response.parsed_output is None:
            raise DecisionFormatError(
                f"no parsed output (stop_reason={response.stop_reason})", billed
            )
        return response.parsed_output, {**billed, "model": self.label}


class ScriptedDecider:
    """A deterministic stand-in. Takes a list (or callable) of Decisions."""

    def __init__(self, script, label="scripted"):
        self._script = script
        self.label = label
        self.calls = []

    def __call__(self, system: str, user: str) -> tuple[Decision, dict]:
        self.calls.append((system, user))
        if callable(self._script):
            decision = self._script(len(self.calls) - 1, system, user)
        else:
            decision = self._script[(len(self.calls) - 1) % len(self._script)]
        return decision, {"input_tokens": 0, "output_tokens": 0, "model": self.label}


# ----------------------------------------------------------------------- factory

DEFAULT_MODEL = os.environ.get("WATCH_CATAN_MODEL", "gpt-5.6-luna")


def make_decider(spec: str, effort: Optional[str] = None, cache_writes: bool = True,
                 flex: bool = False, **kwargs) -> Decider:
    """Build a decider from a model spec, picking the provider from the name.

    `cache_writes=False` is for a caller that never sends the same prompt twice,
    such as a match: every write is paid for and none is read. Scripts that
    re-ask a position keep them, because there the hits outweigh the fee.

    `flex=True` asks for the half-price tier where the provider has one, and is
    refused for a model without it rather than silently ignored."""
    provider, model = resolve(spec)
    if flex:
        if not model.startswith(provider.flex_prefixes or ("\0",)):
            raise ValueError(f"{spec} has no flex tier")
        kwargs["service_tier"] = "flex"
        # OpenAI's own examples give flex requests 15 minutes.
        kwargs.setdefault("timeout", 900.0)
    if not cache_writes and provider.cache_off_body and model.startswith(
            provider.cache_off_prefixes):
        kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), **provider.cache_off_body}
    api_key = os.environ.get(provider.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{provider.api_key_env} is not set, and {model} is served by {provider.name}"
        )
    if provider.kind == "anthropic":
        return AnthropicDecider(model, effort=effort, api_key=api_key, label=spec, **kwargs)
    kwargs.setdefault("max_tokens", provider.max_tokens)
    kwargs.setdefault("timeout", provider.timeout)
    return OpenAICompatDecider(
        model, base_url=provider.base_url, api_key=api_key, effort=effort,
        label=spec, schema_enforced=provider.schema_enforced,
        token_param=provider.token_param, **kwargs
    )

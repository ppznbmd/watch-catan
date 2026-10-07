"""The agent: a persona, a gate, and a strict round-trip through the model.

The gate is the whole cost story. Catan is 200+ decisions a game, most of them
forced or trivial. The model is called for the ones where behaviour is visible —
opening placement, what to build, the robber, and every part of a negotiation —
and a cheap policy takes the rest.
"""

from typing import Optional

from catanatron.models.enums import Action, ActionPrompt, ActionType
from catanatron.models.player import Player

from arena.deciders import DecisionFormatError

from arena.prompt import (
    AUTHOR_OFFER,
    build_prompt,
    describe_action,
    render_offer_on_table,
    resources_to_freqdeck,
    takeable_counters,
    validate_offer,
)
from arena.reflex import Reflex


def is_own_offer(state, color) -> bool:
    """Are we being asked to respond to a trade we ourselves proposed?

    Catanatron walks the table for responses excluding whoever just *answered*,
    not whoever *offered* (`apply_reject_trade`), and `State.__init__` shuffles
    the seating — so the proposer gets asked about its own offer unless it
    happens to sit first. Left alone that burns a model call per offer and, worse,
    lets a player accept and then close a trade with itself.
    """
    if state.current_prompt != ActionPrompt.DECIDE_TRADE:
        return False
    trade = state.current_trade
    return trade is not None and state.colors[trade[10]] == color


def stable_order(playable_actions):
    """Put the legal moves in an order that does not change between processes.

    Catanatron builds maritime trades in a `set` of string tuples
    (`inner_maritime_trade_possibilities`), and Python randomises string hashes
    per process. The engine therefore hands out the same moves in a different
    order every run, and since an agent picks a move *by index*, an identical
    seed would otherwise produce a different game in the next process.
    """
    return sorted(playable_actions, key=lambda a: (a.action_type.value, repr(a.value)))


def may_offer(state, playable_actions, talk, color) -> bool:
    """Offering is legal post-roll on your own turn — END_TURN being available
    is the reliable marker — and only while the turn's offer budget lasts."""
    if state.current_prompt != ActionPrompt.PLAY_TURN:
        return False
    if not any(a.action_type == ActionType.END_TURN for a in playable_actions):
        return False
    return talk.offers_left(state.num_turns, color.value) > 0


def not_addressed(state, talk, color) -> bool:
    """Is the offer on the table put to somebody else alone? A counter-offer
    taken up is directed at its author, but the engine still walks the table."""
    if state.current_prompt != ActionPrompt.DECIDE_TRADE:
        return False
    return talk.addressee() not in (None, color.value)


def may_counter(state, color) -> bool:
    """Answering an offer, a seat may name other terms instead. Only reached
    when the seat is called at all — which it is not when REJECT is its only
    legal move — so a seat that cannot meet the ask cannot counter either."""
    return state.current_prompt == ActionPrompt.DECIDE_TRADE and not is_own_offer(state, color)


#: Decisions made on the seat's own turn. The last one's reasoning is shown back
#: to it as its note: that is where a plan gets written down. An answer to
#: somebody else's offer is about that offer, and would overwrite the plan.
OWN_TURN_PROMPTS = {
    ActionPrompt.BUILD_INITIAL_SETTLEMENT,
    ActionPrompt.BUILD_INITIAL_ROAD,
    ActionPrompt.PLAY_TURN,
}


# Prompts where the agent's behaviour is worth paying to observe.
LLM_PROMPTS = {
    ActionPrompt.BUILD_INITIAL_SETTLEMENT,
    ActionPrompt.BUILD_INITIAL_ROAD,
    ActionPrompt.PLAY_TURN,
    ActionPrompt.MOVE_ROBBER,
    ActionPrompt.DECIDE_TRADE,
    ActionPrompt.DECIDE_ACCEPTEES,
    # Which cards to lose is a real choice, and it used to be made at random by
    # the reflex policy on the agent's behalf. Called once per card, because the
    # engine asks for one card at a time; a hand of one resource type is forced
    # and never reaches the model.
    ActionPrompt.DISCARD,
}


class LLMPlayer(Player):
    def __init__(self, color, persona, decider, talk, log=None, is_bot=True):
        super().__init__(color, is_bot)
        self.persona = persona
        self.decider = decider
        self.model = getattr(decider, "label", "unknown")
        self.talk = talk
        self.log = log
        self._reflex = Reflex(color)
        self.calls = 0
        self.fallbacks = 0
        self.malformed = 0
        self.self_offers = 0
        self.usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
        # `state.action_records` length at this seat's last prompt: the dice
        # section shows what came after it. `arena/rebuild.py` recomputes it from
        # the decision and fallback events, so it must move only where those fire.
        self._seen = 0
        # (turn, reasoning) of this seat's last decision on its own turn; moved
        # only on a `decision` event, which is where `arena/rebuild.py` moves it.
        self._note = None

    # ------------------------------------------------------------------ gate

    def _may_offer(self, state, playable_actions) -> bool:
        return may_offer(state, playable_actions, self.talk, self.color)

    def _worth_a_call(self, state, playable_actions) -> bool:
        if len(playable_actions) == 1 and not self._may_offer(state, playable_actions):
            return False
        return state.current_prompt in LLM_PROMPTS

    # ---------------------------------------------------------------- decide

    def decide(self, game, playable_actions):
        state = game.state
        playable_actions = stable_order(playable_actions)

        if is_own_offer(state, self.color):
            # Reject explicitly rather than fall through to the reflex policy,
            # which could pick ACCEPT and trade us with ourselves.
            action = next(
                a for a in playable_actions if a.action_type == ActionType.REJECT_TRADE
            )
            self.self_offers += 1
            self._emit("own_offer_skipped", action=action)
            return action

        if not_addressed(state, self.talk, self.color):
            action = next(
                a for a in playable_actions if a.action_type == ActionType.REJECT_TRADE
            )
            self._emit("not_addressed", action=action)
            return action

        if not self._worth_a_call(state, playable_actions):
            action = self._reflex.decide(game, playable_actions)
            self._emit("reflex", action=action)
            return action

        may_offer = self._may_offer(state, playable_actions)
        counters = takeable_counters(state, self.talk, self.color) if may_offer else []
        countering = may_counter(state, self.color)
        user = build_prompt(game, self.color, playable_actions, self.talk, may_offer,
                            since=self._seen, note=self._note, may_counter=countering)
        self._seen = len(state.action_records)
        prompt = state.current_prompt
        complaint = None

        for attempt in range(2):
            text = user if complaint is None else f"{user}\n\nTHAT DID NOT WORK: {complaint}\nTry again."
            try:
                decision, usage = self.decider(self.persona.system_prompt, text)
            except DecisionFormatError as exc:
                # The model answered, just not usably. Telling it what was wrong
                # is exactly the kind of failure a retry can fix — and with a
                # provider that does not enforce a schema, this is the common case.
                # The provider billed for it either way, so it counts.
                complaint = str(exc)
                self.malformed += 1
                self._count(exc.usage)
                self._emit("malformed", complaint=complaint, attempt=attempt,
                           usage=exc.usage)
                continue
            except Exception as exc:  # network, rate limit, refusal — never fatal
                complaint = f"the call failed: {exc}"
                self._emit("error", detail=str(exc), attempt=attempt)
                break
            self._count(usage)

            action, complaint, extra = self._to_action(
                state, playable_actions, decision, may_offer, counters, countering)
            if action is not None:
                self.talk.record_move(state.num_turns, self.color.value, action,
                                      decision.say, **extra)
                self._emit("decision", action=action, decision=decision, usage=usage,
                           attempt=attempt, **extra)
                if prompt in OWN_TURN_PROMPTS:
                    self._note = (state.num_turns, decision.reasoning)
                return action
            self._emit("invalid", complaint=complaint, decision=decision, attempt=attempt)

        self.fallbacks += 1
        action = self._reflex.decide(game, playable_actions)
        self._emit("fallback", action=action, complaint=complaint)
        return action

    def _count(self, usage):
        """One request that reached the provider, usable or not."""
        self.calls += 1
        for field in ("input_tokens", "cached_input_tokens", "output_tokens"):
            self.usage[field] = self.usage.get(field, 0) + (usage or {}).get(field, 0)
        # Only where the provider reports it: a 0 here would price every write
        # as if it had not happened.
        if "cache_write_input_tokens" in (usage or {}):
            self.usage["cache_write_input_tokens"] = (
                self.usage.get("cache_write_input_tokens", 0)
                + usage["cache_write_input_tokens"])

    # ----------------------------------------------------------- translation

    def _to_action(self, state, playable_actions, decision, may_offer, counters=(),
                   countering=False):
        """Turn a Decision into a legal Action, or explain why it is not one.

        Returns (action, complaint, extra): `extra` is what the table talk and
        the event log need beyond the action — `to` for an offer put to one
        player, `counter` for a rejection that named other terms."""
        if decision.choice == AUTHOR_OFFER:
            give = resources_to_freqdeck(decision.offer_give)
            want = resources_to_freqdeck(decision.offer_want)
            if countering:
                problem = validate_offer(state, self.color, give, want)
                if problem:
                    return None, problem, {}
                # The engine has no counter-offer: to it this is a rejection.
                reject = next(a for a in playable_actions
                              if a.action_type == ActionType.REJECT_TRADE)
                return reject, None, {"counter": [list(give), list(want)]}
            if not may_offer:
                return None, "you cannot put an offer on the table right now", {}
            problem = validate_offer(state, self.color, give, want)
            if problem:
                return None, problem, {}
            return Action(self.color, ActionType.OFFER_TRADE, (*give, *want)), None, {}

        n = len(playable_actions)
        if isinstance(decision.choice, int) and n <= decision.choice < n + len(counters):
            c = counters[decision.choice - n]
            return (Action(self.color, ActionType.OFFER_TRADE, (*c.want, *c.give)), None,
                    {"to": c.color})

        if not isinstance(decision.choice, int) or not (0 <= decision.choice < n):
            return None, (
                f"{decision.choice} is not one of the moves — pick 0 to "
                f"{n + len(counters) - 1}"
                + (f", or {AUTHOR_OFFER} to offer a trade" if may_offer else "")
                + (f", or {AUTHOR_OFFER} to counter" if countering else "")
            ), {}
        return playable_actions[decision.choice], None, {}

    # ------------------------------------------------------------------ logs

    def _emit(self, kind, action=None, decision=None, **extra):
        if self.log is None:
            return
        payload = {
            "color": self.color.value,
            "persona": self.persona.name,
            "model": self.model,
            **extra,
        }
        if action is not None:
            payload["action"] = {
                "type": action.action_type.value,
                "value": action.value,
                "described": describe_action(action),
            }
        if decision is not None:
            payload["reasoning"] = decision.reasoning
            payload["say"] = decision.say
            if decision.thinking:
                payload["thinking"] = decision.thinking
        self.log.emit(kind, **payload)

    def reset_state(self):
        self.calls = 0
        self.fallbacks = 0
        self.malformed = 0
        self.self_offers = 0
        self.usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
        self._seen = 0
        self._note = None

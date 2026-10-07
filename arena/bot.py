"""The baseline bot, taught to answer a trade offer.

Catanatron's `ValueFunctionPlayer` predates player-to-player trading. It scores
each legal move by applying it to a copy of the game and valuing the result, but
accepting an offer moves no cards — the exchange happens only when the proposer
confirms — so ACCEPT and REJECT always score the same, and the tie goes to
whichever the engine lists first: REJECT. In the first eight real matches the bot
could have accepted 338 offers; all 338 were exact ties, and it rejected all 338.
The table had a seat that could not trade, which no human table has.

This values ACCEPT as the trade completed. It judges only its own position, like
the rest of the value function — it never asks whether the trade helps the
proposer more — so it is a self-interested trader, not a careful one. Everything
else is the stock bot.
"""

from catanatron.models.enums import ActionPrompt, ActionType
from catanatron.players.value import ValueFunctionPlayer, get_value_fn
from catanatron.state_functions import player_freqdeck_add, player_freqdeck_subtract


class TradingValuePlayer(ValueFunctionPlayer):
    def __init__(self, color, talk=None, params=None):
        super().__init__(color, params)
        # Only to see who an offer is put to: a counter-offer taken up is put to
        # the player who made it, and the engine still asks everyone.
        self.talk = talk

    def decide(self, game, playable_actions):
        state = game.state
        if state.current_prompt != ActionPrompt.DECIDE_TRADE:
            return super().decide(game, playable_actions)
        reject = next(a for a in playable_actions if a.action_type == ActionType.REJECT_TRADE)
        accept = next((a for a in playable_actions
                       if a.action_type == ActionType.ACCEPT_TRADE), None)
        if accept is None:
            return reject
        if self.talk is not None and self.talk.addressee() not in (None, self.color.value):
            return reject
        value = get_value_fn(self.value_fn_builder_name, self.params.weights)
        trade = state.current_trade
        after = game.copy()
        player_freqdeck_subtract(after.state, self.color, trade[5:10])
        player_freqdeck_add(after.state, self.color, trade[:5])
        return accept if value(after, self.color) > value(game, self.color) else reject

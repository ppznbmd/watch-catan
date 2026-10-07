"""The baseline bot answers trade offers on their merits.

Catanatron's own bot scores ACCEPT and REJECT identically — accepting moves no
cards until the proposer confirms — and takes REJECT by tie-break. In the first
eight real matches that was 338 rejections out of 338 offers it could have
taken. These tests pin the fix, and pin that it stays a fix to trading only.
"""

from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import RESOURCES, ActionPrompt, ActionType
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.value import ValueFunctionPlayer

from arena.bot import TradingValuePlayer
from arena.table_talk import TableTalk


def offer_to_bot(hand, give, want, seed=4):
    """A mid-game position where RED offers the bot (WHITE) `give` for `want`."""
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE),
                 RandomPlayer(Color.WHITE)], seed=seed)
    for _ in range(40):
        game.play_tick()
    state = game.state
    white, red = state.colors.index(Color.WHITE), state.colors.index(Color.RED)
    for r, n in zip(RESOURCES, hand):
        state.player_state[f"P{white}_{r}_IN_HAND"] = n
    for r in RESOURCES:
        state.player_state[f"P{red}_{r}_IN_HAND"] = 5
    state.current_trade = (*give, *want, red)
    state.is_resolving_trade = True
    state.current_player_index = white
    state.current_prompt = ActionPrompt.DECIDE_TRADE
    # The engine caches the legal moves; they were computed for the old prompt.
    game.playable_actions = generate_playable_actions(state)
    return game


def answer(bot, game):
    return bot.decide(game, game.playable_actions).action_type


#                     wood brick sheep wheat ore
SETTLEMENT_BUT_WHEAT = (1, 1, 1, 0, 2)


def test_the_bot_accepts_the_card_that_completes_a_settlement():
    """Giving a spare ore for the missing wheat is the trade any player takes."""
    game = offer_to_bot(SETTLEMENT_BUT_WHEAT, give=(0, 0, 0, 1, 0), want=(0, 0, 0, 0, 1))
    assert answer(TradingValuePlayer(Color.WHITE), game) == ActionType.ACCEPT_TRADE


def test_the_bot_refuses_to_give_up_the_settlement_it_is_holding():
    """Asked for the brick out of a complete settlement hand, for a card it does
    not need, it says no. A bot that accepted everything would be a gift shop,
    not an opponent."""
    complete = (1, 1, 1, 1, 0)
    game = offer_to_bot(complete, give=(0, 0, 0, 0, 1), want=(0, 1, 0, 0, 0))
    assert answer(TradingValuePlayer(Color.WHITE), game) == ActionType.REJECT_TRADE


def test_the_stock_bot_still_rejects_what_the_trading_bot_accepts():
    """The defect this module exists for. If Catanatron ever values trades itself
    this fails, and the wrapper can go."""
    game = offer_to_bot(SETTLEMENT_BUT_WHEAT, give=(0, 0, 0, 1, 0), want=(0, 0, 0, 0, 1))
    assert answer(ValueFunctionPlayer(Color.WHITE), game) == ActionType.REJECT_TRADE


def test_the_bot_rejects_an_offer_put_to_another_player_alone():
    """A counter-offer taken up is put to its author, but the engine asks every
    seat. The bot must not step into a trade negotiated with somebody else."""
    game = offer_to_bot(SETTLEMENT_BUT_WHEAT, give=(0, 0, 0, 1, 0), want=(0, 0, 0, 0, 1))
    talk = TableTalk()
    talk.record_pitch(game.state.num_turns, "RED", (0, 0, 0, 1, 0), (0, 0, 0, 0, 1),
                      "for you only", to="BLUE")
    assert answer(TradingValuePlayer(Color.WHITE, talk), game) == ActionType.REJECT_TRADE


def test_weighing_a_trade_leaves_the_real_game_untouched():
    """The bot values the trade on a copy. Moving cards in the real state would
    hand the bot the resources before anyone confirmed anything."""
    game = offer_to_bot(SETTLEMENT_BUT_WHEAT, give=(0, 0, 0, 1, 0), want=(0, 0, 0, 0, 1))
    before = dict(game.state.player_state)
    TradingValuePlayer(Color.WHITE).decide(game, game.playable_actions)
    assert dict(game.state.player_state) == before

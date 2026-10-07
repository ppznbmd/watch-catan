"""The cheap policy for decisions not worth a model call.

Delegates to Catanatron's own WeightedRandomPlayer rather than inventing a
heuristic — it already knows what a non-stupid rote move looks like.
"""

from catanatron.players.weighted_random import WeightedRandomPlayer


class Reflex:
    """A per-colour wrapper so one instance serves one seat."""

    def __init__(self, color):
        self._inner = WeightedRandomPlayer(color)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        return self._inner.decide(game, playable_actions)

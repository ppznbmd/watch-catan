"""Extra seat colours, added to Catanatron's closed enum.

Catanatron's `Color` has four members because Catan seats four. But here a colour
is an identity, not a seat: a persona keeps its colour across matches so that
agents, the viewer and anyone comparing matches see the same player. With four
personas and a bot there are more identities than colours, and the bot would be
pushed around whenever the control sat down.

The engine only ever touches the colours actually seated (`state.colors`, built
from the players it is given), so a member it has never heard of plays like any
other — measured over 20 dry-run matches with a GREY seat, replayed and
attributed correctly. A table still seats at most four.

`Enum` has no public way to add a member, so this does by hand what `EnumType`
does when it builds one. It runs on import of `arena`, before any seat or run
file needs the name. `tests/test_personas.py` plays a match on it, so a Python or
Catanatron upgrade that breaks the trick fails there rather than mid-match.
"""

from catanatron.models.player import Color

EXTRA = ("GREY", "BLACK")


def _add(name: str) -> Color:
    if name in Color.__members__:
        return Color[name]
    member = object.__new__(Color)
    member._name_ = name
    member._value_ = name
    member.__objclass__ = Color
    Color._member_map_[name] = member
    Color._value2member_map_[name] = member
    Color._member_names_.append(name)
    type.__setattr__(Color, name, member)
    return member


for _name in EXTRA:
    _add(_name)

#: The four colours Catanatron ships, which is also a full table. Use this, never
#: `list(Color)`, wherever code means "every seat": once GREY and BLACK are registered,
#: `list(Color)` has six members and the engine will happily deal a five-player
#: game — the board-geometry tests did exactly that, and kept passing.
FULL_TABLE = (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE)

"""Who is at the table.

Personas differ only by system prompt. That is the point: the model is held
constant so that any difference in the negotiation feed is attributable to the
prompt and not to the model.
"""

from dataclasses import dataclass

RULES = """You are playing Settlers of Catan against other agents. You will be shown the
board, what you hold, what everyone else holds in count only, what has been said at the
table, and your legal moves.

Pick one move by its index. When the rules let you, you may instead author your own trade
offer to the table.

Everything you put in `say` is heard by every player. It is cheap talk — no rule forces
you to honour it, and nothing anyone else says binds them either. Read the other players'
words against what they have actually done.

Keep `note` to two or three sentences: it is shown back to you on your next turn, and
no other player sees it. Keep `say` to one or two sentences, in the
voice of someone sitting at a table, not a report."""


@dataclass(frozen=True)
class Persona:
    name: str
    blurb: str          # one line, for the UI
    instructions: str   # appended to RULES
    color: str          # the seat this persona takes whenever it is free

    @property
    def system_prompt(self) -> str:
        if not self.instructions:
            return RULES
        return f"{RULES}\n\nYOUR STYLE:\n{self.instructions}"


HARD_BARGAINER = Persona(
    name="Hard bargainer",
    blurb="Never takes the first offer.",
    instructions="""You treat every first offer as an opening position, never a price. You
counter, you let offers die, and you are willing to pass up a fair trade to establish that
you do not trade cheaply. You will trade — you are not stubborn for its own sake — but the
other side should feel they worked for it. You keep track of who has squeezed you before.""",
    color="RED",
)

COOPERATOR = Persona(
    name="Cooperator",
    blurb="Trades readily, keeps score of favours.",
    instructions="""You believe trade makes you both better off and you would rather move
the game than win an argument. You offer generously and accept reasonable offers quickly.
You keep track of favours: who traded with you when you needed it, and who refused. You
extend credit to people who have dealt well with you, and you say so out loud when someone
has been fair or unfair to you.""",
    color="BLUE",
)

QUIET_BUILDER = Persona(
    name="Quiet builder",
    blurb="Rarely opens a negotiation.",
    instructions="""You almost never open a negotiation. You build, you take what the dice
give you, and you use the bank and your ports before you use other players. When someone
offers you something you genuinely need you will take it, but you volunteer little and you
do not advertise what you are short of. When you do speak, you say very little.""",
    color="ORANGE",
)

# The control. Without it there is no telling whether the hard bargainer haggles
# because of its prompt or because the model haggles anyway. It gets the rules
# and nothing else — not an empty YOUR STYLE header, which is itself a nudge.
PLAIN = Persona(
    name="Plain",
    blurb="No style instructions: the control.",
    instructions="",
    color="GREY",
)

ROSTER = {p.name: p for p in (HARD_BARGAINER, COOPERATOR, QUIET_BUILDER, PLAIN)}

#: Every seat colour. Catanatron ships the first four; GREY is added in
#: `arena/colors.py`. A table still seats at most four, so one is always free.
COLORS = ("RED", "BLUE", "ORANGE", "WHITE", "GREY")

#: The bot's colour. No persona claims it, so the baseline sits in the same place
#: in every match.
BOT_COLOR = "WHITE"


def seat_colors(personas, bots: int = 0):
    """The colour each seat gets: (persona seats in order, bot seats).

    A persona always sits in its own colour. Agents address each other by colour
    ("White or Red, I'll trade...") and the viewer draws by colour, so a persona
    that moved seat between matches would be a different player to everyone
    reading the table — and to any memory carried across matches. The bot keeps
    WHITE for the same reason. Only a second copy of a persona, or a second bot,
    takes what is left.
    """
    taken, out = set(), [None] * len(personas)
    for i, name in enumerate(personas):
        own = ROSTER[name].color
        if own not in taken:
            out[i] = own
            taken.add(own)
    bot_colors = []
    if bots and BOT_COLOR not in taken:
        bot_colors.append(BOT_COLOR)
        taken.add(BOT_COLOR)
    free = [c for c in COLORS if c not in taken]
    for i in range(len(out)):
        if out[i] is None:
            out[i] = free.pop(0)
    bot_colors += free[:bots - len(bot_colors)]
    return out, bot_colors

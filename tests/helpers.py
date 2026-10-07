"""Scripted agents: everything below runs without spending a token."""

import re

from arena.deciders import Decision


def index_of(user: str, phrase: str) -> int:
    """Find the index of a legal move by its rendered description.

    Tests pick moves the way a reader would ("end your turn"), so they do not
    break when the engine reorders its action list.
    """
    for line in user.splitlines():
        m = re.match(r"\s*\[(\d+)\]\s+(.*)", line)
        if m and phrase in m.group(2):
            return int(m.group(1))
    raise AssertionError(f"no legal move matching {phrase!r} in:\n{user}")


def can_offer(user: str) -> bool:
    return "write your own trade offer" in user


def offer_on_table(user: str) -> bool:
    return "ON THE TABLE:" in user


def d(choice, reasoning="scripted", say="", give=(), want=()):
    return Decision(
        reasoning=reasoning,
        choice=choice,
        offer_give=list(give),
        offer_want=list(want),
        say=say,
    )

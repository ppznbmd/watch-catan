"""Stated intention against what the seat then built, over a set of matches.

A decision's reasoning is classified by the build it names (city, settlement,
road, development card). The outcome is whether that seat then made that build
within a window of its own turns. The control is the same rate after decisions
of the same seat that did NOT name that build: without it, "they built a road
after saying road" says nothing, because they build roads all the time.

    .venv/bin/python scripts/v3_intent.py runs/20260918-111838-3c062d.jsonl ...

Read-only. Pass files explicitly, one prompt version at a time. From v3 an agent
rereads its own last note, so persistence is partly by construction there.
"""
import collections
import json
import random
import re
import sys

FILES = sys.argv[1:]
if not FILES:
    sys.exit(__doc__)
# The target must follow a word of intention, and not belong to someone else:
# the loose version read "Red's city" and "I cannot buy a development card" as plans.
_INTENT = r"\b(?:need|needs|needed|for|toward|towards|build|complete|finish|afford|reach|buy|set up|assemble|into|next|another|second)\b[^.;]{0,30}?"
_NOT_MINE = "".join(rf"(?<!\b{w} )" for w in ("their", "his", "her", "Red\u2019s", "Blue\u2019s",
                    "Orange\u2019s", "White\u2019s", "Red's", "Blue's", "Orange's", "White's"))
TARGETS = {
    "city": re.compile(_INTENT + _NOT_MINE + r"\bcity\b", re.I),
    "settlement": re.compile(_INTENT + _NOT_MINE + r"\bsettlement\b", re.I),
    "road": re.compile(_INTENT + _NOT_MINE + r"\broad\b(?![- ]building)", re.I),
    "dev": re.compile(_INTENT + _NOT_MINE + r"\b(?:development|dev) card\b", re.I),
}
ACQUIRE = {"OFFER_TRADE", "ACCEPT_TRADE", "CONFIRM_TRADE", "MARITIME_TRADE"}
BUILD = {"BUILD_CITY": "city", "BUILD_SETTLEMENT": "settlement",
         "BUILD_ROAD": "road", "BUY_DEVELOPMENT_CARD": "dev"}
WINDOWS = (0, 1, 3)  # this own turn only; + next own turn; + next three

records = []  # one per decision considered
for f in FILES:
    ev = [json.loads(l) for l in open(f) if l.strip()]
    name = {s["color"]: s["name"] for s in ev[0]["seats"]}
    # per colour: ordered list of (turn, build kind) and the list of own turns
    builds = collections.defaultdict(list)
    own_turns = collections.defaultdict(list)
    owner = {}
    state = None
    decisions = []
    for e in ev:
        if e["kind"] == "state":
            state = e
            # current_color is whoever is asked: the responder to an offer, the
            # discarder on a seven. Only PLAY_TURN names the owner of the turn.
            if e["prompt"] == "PLAY_TURN":
                owner[e["turn"]] = c = e["current_color"]
                if not own_turns[c] or own_turns[c][-1] != e["turn"]:
                    own_turns[c].append(e["turn"])
            continue
        if e["kind"] not in ("decision", "bot", "reflex"):
            continue
        t = e["action"]["type"]
        if state is None or state["prompt"].startswith("BUILD_INITIAL"):
            continue
        if t in BUILD:
            builds[e["color"]].append((state["turn"], BUILD[t]))
        if e["kind"] == "decision":
            decisions.append((state["turn"], e))

    for turn, e in decisions:
        current = owner.get(turn)
        c, t = e["color"], e["action"]["type"]
        if t in BUILD:
            continue  # the build itself is its own fulfilment, not a test
        if t not in ACQUIRE:
            continue  # a trade has a purpose; a rejection or an end turn often names what it cannot do
        r = e["reasoning"]
        hits = {k: m.end() for k, rx in TARGETS.items() if (m := rx.search(r))}  # where the target word is
        stated = min(hits, key=hits.get) if hits else None  # the first one named
        # the window starts at this turn if it is the seat's own, else at its next one
        mine = own_turns[c]
        later = [x for x in mine if x >= turn]
        if c != current:
            later = [x for x in mine if x > turn]
        out = {}
        for w in WINDOWS:
            span = later[: w + 1]
            if len(span) < w + 1:
                out[w] = None  # match ended inside the window
                continue
            lo, hi = (turn, span[-1])
            done = {k for (bt, k) in builds[c] if lo <= bt <= hi
                    and not (bt == turn and c != current)}
            out[w] = done
        records.append(dict(persona=name[c], stated=stated, hits=hits, kind=t,
                            own=c == current, out=out, reasoning=r,
                            described=e["action"]["described"], file=f[-28:], turn=turn))


def rate(rows, target, w):
    rows = [x for x in rows if x["out"][w] is not None]
    if not rows:
        return "   -   "
    k = sum(target in x["out"][w] for x in rows)
    return f"{100 * k / len(rows):3.0f}% ({len(rows)})"


print(f"decisions considered: {len(records)}; with a stated target: "
      f"{sum(r['stated'] is not None for r in records)}; multi-target: "
      f"{sum(len(r['hits']) > 1 for r in records)}")
print("counts of stated target by persona:")
for p in ("Hard bargainer", "Cooperator", "Quiet builder"):
    print("  ", p, collections.Counter(r["stated"] for r in records if r["persona"] == p))

for scope, pred in (("acquiring moves", lambda r: True),
                    ("on own turn", lambda r: r["own"]),
                    ("answering someone else's offer", lambda r: not r["own"])):
    print(f"\n== {scope}: built X within window | stated X  vs  did not state X")
    for p in ("Hard bargainer", "Cooperator", "Quiet builder", None):
        rows = [r for r in records if pred(r) and (p is None or r["persona"] == p)]
        for target in TARGETS:
            said = [r for r in rows if r["stated"] == target]
            not_said = [r for r in rows if target not in r["hits"]]
            cells = "  ".join(f"w{w}: {rate(said, target, w)} vs {rate(not_said, target, w)}"
                              for w in WINDOWS)
            print(f"  {(p or 'ALL'):15s} {target:10s} {cells}")

# what they built instead, when a stated city/settlement did not happen by next own turn
print("\n== stated target not built by next own turn: what was built instead (w1)")
for target in TARGETS:
    miss = [r for r in records if r["stated"] == target and r["out"][1] is not None
            and target not in r["out"][1]]
    inst = collections.Counter(k for r in miss for k in (r["out"][1] or {"nothing"}))
    nothing = sum(1 for r in miss if not r["out"][1])
    print(f"  {target:10s} missed {len(miss)}: built nothing in {nothing}; also built {dict(inst)}")

# Read these by hand: a regex over free text is only as good as its last check.
random.seed(7)
print("\n== spot check of the classification")
for r in random.sample([r for r in records if r["stated"]], 30):
    print(f"  [{r['stated']}] {r['persona']} | {r['described']} | {r['reasoning'][:260]}")
print("\n== spot check of decisions with no target found")
for r in random.sample([r for r in records if not r["stated"]], 8):
    print(f"  {r['persona']} | {r['described']} | {r['reasoning'][:200]}")

# Persistence: the target named on one own turn against the next own turn that names one.
print("\n== persistence: target named on an own turn, same as the next own turn that names one?")
by_seat = collections.defaultdict(list)
for r in records:
    if r["own"] and r["stated"]:
        by_seat[(r["file"], r["persona"])].append((r["turn"], r["stated"]))
for p in ("Hard bargainer", "Cooperator", "Quiet builder"):
    same = total = 0
    for (f, q), seq in by_seat.items():
        if q != p:
            continue
        per_turn = {}
        for turn, st in seq:
            per_turn.setdefault(turn, st)  # first target named in that turn
        turns = sorted(per_turn)
        for a, b in zip(turns, turns[1:]):
            total += 1
            same += per_turn[a] == per_turn[b]
    print(f"  {p:15s} same target next time: {same}/{total} = {100*same/max(total,1):.0f}%")

# One observation per (seat, own turn): repeated offers in a turn are not
# independent evidence. "Stated X" = named X in any acquiring move that turn.
print("\n== deduplicated, own turns only: built X by next own turn (w1) / within three (w3)")
turns = {}
for r in records:
    if not r["own"]:
        continue
    key = (r["file"], r["persona"], r["turn"])
    t = turns.setdefault(key, dict(persona=r["persona"], hits=set(), out=r["out"]))
    t["hits"] |= set(r["hits"])
for p in ("Hard bargainer", "Cooperator", "Quiet builder", None):
    rows = [t for t in turns.values() if p is None or t["persona"] == p]
    for target in TARGETS:
        cells = []
        for w in (1, 3):
            said = [t for t in rows if target in t["hits"] and t["out"][w] is not None]
            other = [t for t in rows if target not in t["hits"] and t["out"][w] is not None]
            a = sum(target in t["out"][w] for t in said)
            b = sum(target in t["out"][w] for t in other)
            cells.append(f"w{w}: {a}/{len(said)} = {100*a/max(len(said),1):3.0f}%  vs  {100*b/max(len(other),1):3.0f}% ({len(other)})")
        print(f"  {(p or 'ALL'):15s} {target:10s} " + "   ".join(cells))

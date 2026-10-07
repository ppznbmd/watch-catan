"""What each seat did, summed over a set of matches: builds, bank and port
trades, development cards, robber, offers and answers.

    .venv/bin/python scripts/v3_actions.py runs/20260918-111838-3c062d.jsonl ...

Read-only. Pass files explicitly; it does not glob, for the same reason as
scripts/offers.py. Pass one prompt version at a time: never pool v1, v2 and v3.
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arena.offers import offers  # noqa: E402

FILES = sys.argv[1:]
if not FILES:
    sys.exit(__doc__)
ACTORS = ("decision", "bot", "reflex", "own_offer_skipped", "not_addressed")

T = collections.defaultdict(collections.Counter)
for f in FILES:
    ev = [json.loads(l) for l in open(f) if l.strip()]
    gs, go = ev[0], ev[-1]
    name = {s["color"]: s["name"] for s in gs["seats"]}
    for c in name:
        T[name[c]]["matches"] += 1
    T[name[go["winner"]]]["wins"] += 1

    settlements, roads = collections.Counter(), collections.Counter()
    free_roads = collections.Counter()
    last_action = None
    for e in ev:
        if e["kind"] == "invalid":
            T[name[e["color"]]]["invalid"] += 1
        if e["kind"] not in ACTORS:
            continue
        c, a = e["color"], e["action"]
        p, t = name[c], a["type"]
        s = T[p]
        if t == "ROLL":
            s["rolls"] += 1
        elif t == "BUILD_SETTLEMENT":
            settlements[c] += 1
            s["settlement_initial" if settlements[c] <= 2 else "settlement"] += 1
        elif t == "BUILD_ROAD":
            roads[c] += 1
            if roads[c] <= 2:
                s["road_initial"] += 1
            elif free_roads[c]:
                free_roads[c] -= 1
                s["road_free"] += 1
            else:
                s["road"] += 1
        elif t == "BUILD_CITY":
            s["city"] += 1
        elif t == "MARITIME_TRADE":
            n = a["described"].split()[1]
            s[{"4": "bank_4to1", "3": "port_3to1", "2": "port_2to1"}[n]] += 1
        elif t == "BUY_DEVELOPMENT_CARD":
            s["dev_bought"] += 1
        elif t == "PLAY_KNIGHT_CARD":
            s["knight"] += 1
        elif t == "PLAY_MONOPOLY":
            s["monopoly"] += 1
        elif t == "PLAY_YEAR_OF_PLENTY":
            s["year_of_plenty"] += 1
        elif t == "PLAY_ROAD_BUILDING":
            s["road_building"] += 1
            free_roads[c] += 2
        elif t == "MOVE_ROBBER":
            s["robber_after_knight" if last_action == (c, "PLAY_KNIGHT_CARD") else "robber_after_7"] += 1
        elif t == "DISCARD_RESOURCE":
            s["discard_events"] += 1
        elif t == "ACCEPT_TRADE":
            s["accept_" + e["kind"]] += 1
        elif t == "REJECT_TRADE":
            if e.get("counter"):
                s["counter_made"] += 1
            else:
                s["reject_" + e["kind"]] += 1
        last_action = (c, t)

    for o in offers(ev):
        s = T[name[o.proposer]]
        s["offers"] += 1
        s["offers_to_one"] += o.to is not None
        s["offers_confirmed"] += o.closed == "confirmed"
        s["counters_received"] += len(o.counters)
        s["counter_taken_up"] += o.to is not None
        s["counter_taken_up_traded"] += o.to is not None and o.closed == "confirmed"
        s["offer_nobody_accepted"] += not o.accepted_by()
        s["offer_bot_accepted"] += any(name[c] == "TradingValuePlayer" for c in o.accepted_by())
        if o.closed == "confirmed":
            T[name[o.partner]]["trades_as_responder"] += 1
            T[name[o.partner]]["partner_of_" + name[o.proposer]] += 1

    final = next(e for e in reversed(ev) if e["kind"] == "state")
    ps = final["player_state"]
    for i, c in enumerate(gs["order"]):
        s = T[name[c]]
        s["vp"] += ps[f"P{i}_ACTUAL_VICTORY_POINTS"]
        s["longest_road"] += ps[f"P{i}_HAS_ROAD"]
        s["largest_army"] += ps[f"P{i}_HAS_ARMY"]
        s["vp_dev_cards"] += ps.get(f"P{i}_VICTORY_POINT_IN_HAND", 0)

print(json.dumps({p: dict(c) for p, c in T.items()}, indent=1))

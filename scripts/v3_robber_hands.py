"""Robber targeting, hand sizes at the roll, and who a closed trade helps.

    .venv/bin/python scripts/v3_robber_hands.py runs/20260918-111838-3c062d.jsonl ...

Read-only. Pass files explicitly, one prompt version at a time. The leader is
read from the public score: the state's own victory_points counts hidden cards.
"""
import collections
import json
import re
import sys

FILES = sys.argv[1:]
if not FILES:
    sys.exit(__doc__)
RES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
BUILDS = {"BUILD_CITY", "BUILD_SETTLEMENT", "BUILD_ROAD", "BUY_DEVELOPMENT_CARD"}
C = collections.defaultdict(collections.Counter)
hand_at_roll = collections.defaultdict(list)

for f in FILES:
    ev = [json.loads(l) for l in open(f) if l.strip()]
    gs = ev[0]
    name = {s["color"]: s["name"] for s in gs["seats"]}
    idx = {c: i for i, c in enumerate(gs["order"])}
    state, owner = None, {}
    built_in_turn = collections.defaultdict(set)   # turn -> colours that built
    trades = []  # (turn, proposer, partner)
    for e in ev:
        if e["kind"] == "state":
            state = e
            if e["prompt"] == "PLAY_TURN":
                owner[e["turn"]] = e["current_color"]
            continue
        if e["kind"] not in ("decision", "bot", "reflex") or state is None:
            continue
        if state["prompt"].startswith("BUILD_INITIAL"):
            continue
        a, c = e["action"], e["color"]
        t = a["type"]
        if t in BUILDS:
            built_in_turn[state["turn"]].add(c)
        if t == "ROLL":
            ps = state["player_state"]
            hand_at_roll[name[c]].append(sum(ps[f"P{idx[c]}_{r}_IN_HAND"] for r in RES))
        if t == "MOVE_ROBBER":
            # the public score: state["victory_points"] includes hidden VP cards nobody else sees
            vp = {x: state["player_state"][f"P{idx[x]}_VICTORY_POINTS"] for x in idx}
            m = re.search(r"stealing from (\w+)", a["described"])
            victim = m.group(1) if m and m.group(1) in name else None
            others = [x for x in vp if x != c]
            top = max(vp[x] for x in others)
            leaders = [x for x in others if vp[x] == top]
            s = C[name[c]]
            s["robber_moves"] += 1
            s["robber_no_victim"] += victim is None
            if victim:
                s["robber_hit_top_opponent"] += victim in leaders
                s["robber_expected_top"] += len(leaders) / len(others)
                s["robber_hit_bot"] += name[victim] == "TradingValuePlayer"
                C["victims"][name[victim]] += 1
        if t == "CONFIRM_TRADE":
            partner = str(a["value"][-1]).rsplit(".", 1)[-1]
            trades.append((state["turn"], c, partner))

    # a turn counts once per seat for the baseline
    turns_by = collections.defaultdict(list)
    for turn, c in owner.items():
        turns_by[c].append(turn)
    traded_turns = {(t, p) for t, p, _ in trades}
    for c, ts in turns_by.items():
        for t in ts:
            k = "turn_with_own_trade" if (t, c) in traded_turns else "turn_without"
            C[name[c]][k] += 1
            C[name[c]][k + "_built"] += c in built_in_turn[t]
    for t, p, q in trades:
        nxt = [x for x in turns_by[q] if x > t]
        if nxt:
            C[name[q]]["responded_trade"] += 1
            C[name[q]]["responded_trade_built_next_turn"] += q in built_in_turn[nxt[0]]
    # responder baseline: its own turns that follow a turn in which it did not close a trade as partner
    partner_turns = {(t, q) for t, _, q in trades}
    for c, ts in turns_by.items():
        prev = None
        for t in sorted(ts):
            if prev is not None:
                had = any((x, c) in partner_turns for x in range(prev, t))
                if not had:
                    C[name[c]]["own_turn_no_prior_trade"] += 1
                    C[name[c]]["own_turn_no_prior_trade_built"] += c in built_in_turn[t]
            prev = t

for p, h in hand_at_roll.items():
    h = sorted(h)
    print(f"{p:18s} hand at own roll: mean {sum(h)/len(h):.1f}  median {h[len(h)//2]}  "
          f"over 7: {sum(x > 7 for x in h)}/{len(h)}")
for p, c in C.items():
    print(p, json.dumps(dict(sorted(c.items()))))

"""Table talk and negotiation measures over a set of matches: terms of closed
trades, pitches against terms, answers to the leader, named addressees,
counter-offers taken up and whether their author honoured them.

    .venv/bin/python scripts/v3_talk.py runs/20260918-111838-3c062d.jsonl ...

Read-only. Pass files explicitly, one prompt version at a time.
"""
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arena.offers import offers  # noqa: E402

FILES = sys.argv[1:]
if not FILES:
    sys.exit(__doc__)
RES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
COLORS = ["RED", "BLUE", "ORANGE", "WHITE"]

C = collections.defaultdict(collections.Counter)
samples = collections.defaultdict(list)

for f in FILES:
    ev = [json.loads(l) for l in open(f) if l.strip()]
    gs = ev[0]
    name = {s["color"]: s["name"] for s in gs["seats"]}
    idx = {c: i for i, c in enumerate(gs["order"])}
    ps = None
    cur_offer = None  # (proposer, give, want, first_in_turn)
    offers_this_turn = collections.Counter()
    turn = None
    accepted_before = collections.defaultdict(set)  # proposer -> who accepted its offers so far
    for e in ev:
        k = e["kind"]
        if k == "state":
            ps = e["player_state"]
            if e["turn"] != turn:
                turn, offers_this_turn = e["turn"], collections.Counter()
            continue
        if k not in ("decision", "bot", "reflex"):
            continue
        a, c = e["action"], e["color"]
        t = a["type"]
        vp = {x: ps[f"P{idx[x]}_VICTORY_POINTS"] for x in idx} if ps else {}
        if t == "OFFER_TRADE":
            v = a["value"]
            offers_this_turn[c] += 1
            cur_offer = dict(proposer=c, give=sum(v[:5]), want=sum(v[5:10]),
                             nth=offers_this_turn[c], say=e.get("say") or "")
            s = C[name[c]]
            ratio = "even" if cur_offer["give"] == cur_offer["want"] else (
                "gives more" if cur_offer["give"] > cur_offer["want"] else "asks more")
            s["offer_" + ratio] += 1
            say = (e.get("say") or "").lower()
            if re.search(r"\bfair\b|generous|good deal|strong deal|solid deal|great deal", say):
                s["pitch_fair_" + ratio] += 1
                if ratio == "asks more":
                    samples["pitch_fair_asks_more"].append((name[c], a["described"], e["say"]))
            named = [x for x in COLORS if x != c and re.search(r"\b" + x.lower() + r"\b", say)]
            cur_offer["named"] = named
            if named:
                s["offer_names_someone"] += 1
            if k == "decision" and re.search(r"\bI(?:'|’)ll (?:accept|take your)\b|\baccept (?:his|her|their|RED|BLUE|ORANGE|WHITE)", e["reasoning"]):
                samples["offer_reasoned_as_accept"].append((name[c], a["described"], e["reasoning"][:200]))
            continue
        if t in ("ACCEPT_TRADE", "REJECT_TRADE") and cur_offer and k in ("decision", "bot"):
            p = cur_offer["proposer"]
            ans = "accept" if t == "ACCEPT_TRADE" else ("counter" if e.get("counter") else "reject")
            s = C[name[c]]
            # leader: proposer strictly ahead of every other seat on public points
            lead = vp and all(vp[p] > vp[x] for x in vp if x != p)
            s[f"resp_{'leader' if lead else 'nonleader'}_{ans}"] += 1
            if k == "decision":
                s[f"resp_nth{min(cur_offer['nth'], 2)}_{ans}"] += 1
                if c in cur_offer.get("named", []):
                    s[f"resp_named_{ans}"] += 1
                else:
                    s[f"resp_unnamed_{ans}"] += 1
                # reciprocity: did this proposer accept one of my offers earlier in the match?
                rec = "owed" if p in accepted_before[c] else "not_owed"
                s[f"resp_{rec}_{ans}"] += 1
                r = e["reasoning"].lower()
                if ans != "accept" and re.search(r"lead|ahead|closest to winning|points", r):
                    s["reject_mentions_leader"] += 1
                if ans != "accept":
                    s["reject_chosen"] += 1
            if ans == "accept":
                accepted_before[p].add(c)
            continue

    for o in offers(ev):
        if o.to:
            ans = o.answers.get(o.to, ("?", "?"))
            C["counters"][f"author_answered_{ans[0]}_{ans[1]}"] += 1
            samples["counter_taken_up"].append((name[o.proposer], name[o.to], o.describe(), ans, o.closed))
        if o.closed == "confirmed":
            g, w = sum(o.give), sum(o.want)
            C[name[o.proposer]]["traded_cards_given"] += g
            C[name[o.proposer]]["traded_cards_got"] += w
            C[name[o.partner]]["resp_traded_cards_given"] += w
            C[name[o.partner]]["resp_traded_cards_got"] += g

for p, c in C.items():
    print(p, json.dumps(dict(sorted(c.items()))))
for k, v in samples.items():
    print("\n==", k, len(v))
    for x in v[:12]:
        print("  ", x)

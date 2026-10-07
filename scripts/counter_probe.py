"""Does an agent give way against its own word on price?

52 offers in the v3 matches came with a price claim ("I'm not paying a
premium", "firm", "last call", "not a giveaway"). Four got a counter, all
refused, and none of the four offered worse terms for the same cards: the claim
was never put to the test. This puts it there, on positions already played.

A position is the proposer's next decision in the same turn, after an offer of
its own that nobody took and nobody countered: the moment a counter would have
arrived. `arena/rebuild.py` puts it back exactly, and one counter is added to
the table talk, in place of its author's reply to the offer:

  worse  the same card, for one card more than the proposer offered — the
         premium it said it would not pay. The card is more of what it offered
         if it holds one, else another card it holds
  same   the same card, for exactly what the proposer offered: how readily it
         takes up a counter at all

Two groups of positions: `claimed`, the offer came with a price claim; and
`unclaimed`, an equal number drawn from offers without one. If the claim holds
the price, `claimed` takes up `worse` less often than `unclaimed` does.

    .venv/bin/python scripts/counter_probe.py --dry-run runs/<match>.jsonl ...  # free
    .venv/bin/python scripts/counter_probe.py --measure runs/<match>.jsonl ...  # 2 calls
    .venv/bin/python scripts/counter_probe.py runs/<match>.jsonl ...            # the batch

Answers go to experiments/, one line each as they arrive. Nothing here writes
to runs/.
"""

import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catanatron.models.enums import ActionPrompt, ActionType  # noqa: E402
from catanatron.models.decks import freqdeck_subtract  # noqa: E402
from catanatron.state_functions import get_player_freqdeck, player_freqdeck_add  # noqa: E402

from arena.deciders import DecisionFormatError, ScriptedDecider, make_decider  # noqa: E402
from arena.env import load_env  # noqa: E402
from arena.play import usd  # noqa: E402
from arena.prompt import AUTHOR_OFFER, describe_action, describe_counter, takeable_counters  # noqa: E402
from arena.rebuild import detach, plies  # noqa: E402
from arena.scripted import heuristic_script  # noqa: E402
from arena.table_talk import Utterance  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "experiments"
RES = ["wood", "brick", "sheep", "wheat", "ore"]
PRICE = re.compile(
    r"\b(last|final) (offer|call|price)\b|\bfirm\b|\bmy floor\b|\bnot (improving|sweetening|budging|"
    r"going (lower|higher)|discounting|adding anything|paying (a |any )?(premium|extra)|"
    r"adding (a |any )?premium)\b|\bwon['’]t (sweeten|improve|go higher)\b|\bbest offer\b"
    r"|\b(remains|stands)\b|\bno premium\b|\bnot giving (these|them|it) away\b|\bnot a giveaway\b", re.I)
CONDITIONS = ("worse", "same")


def cards(deck):
    parts = [f"{n} {RES[i]}" for i, n in enumerate(deck) if n]
    return " and ".join(parts) if parts else "nothing"


def find_positions(paths):
    """(ply, offer) for each first offer of a turn that nobody took or countered,
    at the proposer's next decision in that turn."""
    out = []
    for path in paths:
        pending = None
        for ply in plies(path):
            state = ply.game.state
            a = ply.actor or {}
            if pending and state.num_turns != pending["turn"]:
                pending = None
            if pending and ply.action.color.value != pending["color"]:
                if a.get("counter"):
                    pending["countered"] = True
                if ply.action.action_type == ActionType.ACCEPT_TRADE:
                    pending["accepted"] = True
                continue
            if pending and a.get("kind") == "decision" and state.current_prompt == ActionPrompt.PLAY_TURN:
                if not pending["countered"] and not pending["accepted"]:
                    out.append((detach(ply), pending))
                pending = None
            if (a.get("kind") == "decision" and ply.action.action_type == ActionType.OFFER_TRADE
                    and ply.talk.offers_left(state.num_turns, ply.action.color.value)
                    == ply.talk.max_offers_per_turn):
                v = ply.action.value
                pending = {"run": Path(path).name, "seq": a["seq"], "turn": state.num_turns,
                           "color": ply.action.color.value, "give": tuple(v[:5]), "want": tuple(v[5:10]),
                           "say": a.get("say") or "", "claimed": bool(PRICE.search(a.get("say") or "")),
                           "countered": False, "accepted": False}
    return out


def premium(offer, mine):
    """What the proposer is asked in `worse`: its own offer plus one card. The
    same resource as its largest if it holds one more, else whichever other card
    it holds most of. None if it holds nothing beyond its offer."""
    give, want = list(offer["give"]), offer["want"]
    order = [max(range(5), key=lambda i: give[i])] + sorted(
        (i for i in range(5) if not want[i]), key=lambda i: -(mine[i] - give[i]))
    for i in order:
        if mine[i] > give[i]:
            give[i] += 1
            return give
    return None


def inject(ply, offer, condition):
    """Add one counter to `ply.talk` and return it, or None if the proposer
    could not meet it. Mutates `ply.game`: pass a detached ply.

    Most offers that nobody took were offers nobody could take: no other seat
    held the card. Leaving those out would keep only the offers that failed for
    other reasons, so the counter's author is dealt the missing cards from the
    bank. Its card count, the only part of its hand the proposer sees, goes up
    by what it was dealt. Authors are agents before the bot, which never counters."""
    state = ply.game.state
    me = ply.action.color
    mine = get_player_freqdeck(state, me)
    ask = premium(offer, mine) if condition == "worse" else list(offer["give"])
    if ask is None or any(h < w for h, w in zip(mine, ask)):
        return None
    llm = {s["color"] for s in ply.start["seats"] if s.get("model")}
    others = sorted((c for c in state.colors if c != me), key=lambda c: c.value not in llm)
    holds = lambda c: all(h >= w for h, w in zip(get_player_freqdeck(state, c), offer["want"]))
    author = next((c for c in others if holds(c)), others[0])
    missing = [max(0, w - h) for h, w in zip(get_player_freqdeck(state, author), offer["want"])]
    if any(m > b for m, b in zip(missing, state.resource_freqdeck)):
        return None
    if any(missing):
        state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, missing)
        player_freqdeck_add(state, author, missing)
    text = f"I'll give you {cards(offer['want'])} for {cards(ask)}. That's my price."
    u = Utterance(state.num_turns, author.value, "counter", text, tuple(offer["want"]), tuple(ask), me.value)
    # A counter is its author's answer to the offer, not a second one: it takes
    # the place of whatever the author replied, or "I'm holding the wood, pass"
    # would sit right above "I'll give you the wood".
    talk = ply.talk.utterances
    pitch = max(i for i, x in enumerate(talk) if x.kind == "pitch" and x.color == me.value)
    reply = next((i for i in range(pitch + 1, len(talk))
                  if talk[i].color == author.value and talk[i].kind == "reply"), None)
    if reply is None:
        talk.append(u)
    else:
        talk[reply] = u
    return u if u in takeable_counters(state, ply.talk, me) else None


def ask(decider, ply, offer, condition):
    counter = inject(ply, offer, condition)
    if counter is None:
        return None
    system, user = ply.prompt()
    legal = ply.legal()
    counters = takeable_counters(ply.game.state, ply.talk, ply.action.color)
    started = time.time()
    try:
        decision, usage = decider(system, user)
    except DecisionFormatError as exc:
        return {"unusable": str(exc), "usage": exc.usage}
    except Exception as exc:  # transport: counted, never fatal to the batch
        return {"error": str(exc)}
    n = len(legal)
    if decision.choice == AUTHOR_OFFER:
        described, took = f"offer {decision.offer_give} for {decision.offer_want}", False
    elif isinstance(decision.choice, int) and 0 <= decision.choice < n:
        described, took = describe_action(legal[decision.choice]), False
    elif isinstance(decision.choice, int) and n <= decision.choice < n + len(counters):
        c = counters[decision.choice - n]
        described, took = describe_counter(c), c is counter
    else:
        return {"unusable": f"index {decision.choice} out of range", "usage": usage}
    return {"choice": decision.choice, "described": described, "took": took,
            "reoffers": decision.choice == AUTHOR_OFFER, "reasoning": decision.reasoning, "thinking": decision.thinking,
            "say": decision.say, "usage": usage, "seconds": round(time.time() - started, 1),
            "counter": counter.render()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--samples", type=int, default=3, help="answers per position and condition")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="draws the unclaimed group")
    ap.add_argument("--controls", type=int, default=None,
                    help="size of the unclaimed group; as many as claimed by default")
    ap.add_argument("--dry-run", action="store_true", help="scripted answers, spends nothing")
    ap.add_argument("--measure", action="store_true", help="one call per condition, then stop")
    ap.add_argument("--model", default=None, help="ask this model instead of the one that played")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    conditions = args.conditions.split(",")

    found = find_positions(args.runs)
    claimed = [p for p in found if p[1]["claimed"]]
    rest = [p for p in found if not p[1]["claimed"]]
    size = len(claimed) if args.controls is None else args.controls
    unclaimed = random.Random(args.seed).sample(rest, min(len(rest), size))
    # a position counts only if both counters can be put in it, so the two
    # conditions are asked on the same positions
    groups = {g: [(ply, o) for ply, o in items
                  if all(inject_ok(ply, o, c) for c in conditions)]
              for g, items in (("claimed", claimed), ("unclaimed", unclaimed))}
    print(f"offers left alone, then decided on again: {len(found)} "
          f"({len(claimed)} claimed a price); usable: "
          + ", ".join(f"{g} {len(v)}" for g, v in groups.items()))

    if not any(groups.values()):
        print("no usable position: nothing to ask")
        return 1
    jobs = [(g, ply, o, c, k) for g, items in groups.items() for ply, o in items
            for c in conditions for k in range(args.samples)]
    if args.measure:
        g = next(g for g, v in groups.items() if v)
        jobs = [(g, groups[g][0][0], groups[g][0][1], c, 0) for c in conditions]

    if not args.dry_run:
        load_env()
    model = args.model or next(ply.seat["model"] for v in groups.values() for ply, _ in v)
    if args.dry_run:
        decider = ScriptedDecider(heuristic_script("trader", rng=random.Random(1)), label="dry-run")
    else:
        decider = make_decider(model)

    OUT_DIR.mkdir(exist_ok=True)
    tag = "dryrun-" if args.dry_run else ("measure-" if args.measure else "")
    out = args.out or OUT_DIR / f"counter-probe-{tag}{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    lock = threading.Lock()
    rows = []
    with out.open("a", buffering=1) as fh:
        fh.write(json.dumps({"kind": "plan", "runs": [str(p) for p in args.runs], "model": model,
                             "conditions": conditions, "samples": args.samples, "jobs": len(jobs),
                             "groups": {g: [f"{o['run']}:{o['seq']}" for _, o in v]
                                        for g, v in groups.items()}}) + "\n")
        print(f"{len(jobs)} calls -> {out}")

        def run(job):
            g, ply, o, c, k = job
            # each worker needs its own copy: the counter is added to the talk
            answer = ask(decider, detach(ply), o, c)
            row = {"kind": "answer", "group": g, "condition": c, "sample": k, "model": model,
                   "offer": {x: o[x] for x in ("run", "seq", "turn", "color", "give", "want", "say")},
                   "persona": ply.seat["name"], "answer": answer,
                   "usd": None if args.dry_run else usd(answer.get("usage"), model)}
            with lock:
                fh.write(json.dumps(row, default=str) + "\n")
                rows.append(row)
            print(f"  {len(rows)}/{len(jobs)} {g} {c} "
                  f"{'took' if answer.get('took') else answer.get('described', 'UNUSABLE')[:50]}", flush=True)

        with ThreadPoolExecutor(max_workers=1 if args.dry_run else args.workers) as pool:
            for f in as_completed([pool.submit(run, j) for j in jobs]):
                f.result()

    print(f"\n{'group':10s} {'condition':9s} {'answers':>7s} {'took':>6s} {'re-offer':>8s} {'unusable':>8s}")
    for g in groups:
        for c in conditions:
            rs = [r["answer"] for r in rows if r["group"] == g and r["condition"] == c]
            ok = [a for a in rs if "choice" in a]
            if not rs:
                continue
            took = sum(a["took"] for a in ok)
            print(f"{g:10s} {c:9s} {len(ok):7d} {took / len(ok) if ok else 0:6.0%} "
                  f"{sum(a['reoffers'] for a in ok):8d} {len(rs) - len(ok):8d}")
    if args.measure:
        for r in rows:
            u = r["answer"].get("usage") or {}
            print(f"\n{r['condition']}: {u.get('input_tokens')} in, {u.get('output_tokens')} out, "
                  f"{r['answer'].get('seconds')}s, ${r['usd'] or 0:.5f}\n  -> {r['answer'].get('described')}")
    total = sum(r["usd"] or 0 for r in rows)
    if total:
        print(f"\nspent ${total:.4f}")
    print(f"answers: {out}")
    return 0


def inject_ok(ply, offer, condition):
    return inject(detach(ply), offer, condition) is not None


if __name__ == "__main__":
    sys.exit(main())

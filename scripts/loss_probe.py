"""Does a model hand the game to the player who is about to win?

The mirror of `scripts/win_probe.py`. There, a road wins and the question is
whether the model sees it. Here, an offer is on the table from a player on 9
visible points, and accepting it gives that player the one card it lacks to
reach 10 on the spot. Accepting is the one answer to an offer that is wrong
without any argument about strategy: the game ends and the responder has lost.

A real match did this once: RED on 9 offered two wheat for one ore, said "I need
the ore", ORANGE answered "Done." and RED built the winning city
(runs/20260918-140426-6b668e.jsonl, turn 93). One case is an anecdote, so the
positions are made here instead. Four value bots play each other on fresh
seeds, and the first moment of each game is kept where

- the player to move has rolled, holds no development cards, and has no hidden
  victory points, so everyone can see how close it is;
- one card X, bought for one card Y it holds, would let it win with a single
  build, and some other player holds X;
- it cannot win this turn without that trade, not even through two trades with
  the bank or its ports.

The offer "one Y for one X" is then put on the table to a player holding X,
with a short neutral pitch, and the model answers as that player.

Each position has a control on the same board: the latest earlier moment where
the same proposer, on 5 points or fewer, could have made the same offer to the
same responder. The question the pair answers is whether the proposer's score
changes the answer, not just how often the model accepts.

    .venv/bin/python scripts/loss_probe.py generate --games 400
    .venv/bin/python scripts/loss_probe.py ask experiments/loss-positions-<ts>.pkl --dry-run
    .venv/bin/python scripts/loss_probe.py ask experiments/loss-positions-<ts>.pkl --measure
    .venv/bin/python scripts/loss_probe.py ask experiments/loss-positions-<ts>.pkl
    .venv/bin/python scripts/loss_probe.py report experiments/loss-probe-<ts>.jsonl

Nothing here reads or writes runs/.
"""

import argparse
import json
import math
import pickle
import random
import re
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import arena  # noqa: F401,E402  registers the extra seat colours
from catanatron.game import Game  # noqa: E402
from catanatron.models.actions import generate_playable_actions  # noqa: E402
from catanatron.models.enums import RESOURCES, Action, ActionPrompt, ActionType  # noqa: E402
from catanatron.models.player import Color  # noqa: E402
from catanatron.players.value import ValueFunctionPlayer  # noqa: E402
from catanatron.state_functions import get_player_freqdeck  # noqa: E402

from arena.deciders import DecisionFormatError, ScriptedDecider, make_decider  # noqa: E402
from arena.env import load_env  # noqa: E402
from arena.llm_player import may_counter, stable_order  # noqa: E402
from arena.personas import ROSTER  # noqa: E402
from arena.play import usd  # noqa: E402
from arena.prompt import AUTHOR_OFFER, PROMPT_VERSION, build_prompt  # noqa: E402
from arena.scripted import heuristic_script  # noqa: E402
from arena.table_talk import TableTalk  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "experiments"
SEATS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
MAX_OFFERS = 2
MAX_TURNS = 400
FAR_VP = 5
#: Said with every offer, in both conditions. Neutral on purpose: it names the
#: card, as players do, and says nothing about what it is for.
PITCH = "Anyone spare a {want}? I'll give a {give} for it."
#: Moves that can end a game by luck or hidden information. A win the responder
#: could not have seen coming does not count, and neither does one the
#: proposer could have drawn without anybody's help.
UNSEEN = {ActionType.BUY_DEVELOPMENT_CARD, ActionType.ROLL}


def vp(state, color, actual=False):
    i = state.color_to_index[color]
    return state.player_state[f"P{i}_{'ACTUAL_' if actual else ''}VICTORY_POINTS"]


def holds_dev_cards(state, color) -> bool:
    i = state.color_to_index[color]
    return any(state.player_state.get(f"P{i}_{d}_IN_HAND", 0)
               for d in ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING",
                         "VICTORY_POINT"))


def after(game, action):
    g = game.copy()
    g.execute(action, validate_action=False)
    return g


def wins_now(game, color) -> bool:
    """One visible move from here wins."""
    return any(after(game, a).winning_color() == color
               for a in game.playable_actions if a.action_type not in UNSEEN)


def wins_without_help(game, color, bank_trades=2) -> bool:
    """Can win this turn alone: a build, after up to `bank_trades` trades with
    the bank or a port."""
    frontier = [game]
    for depth in range(bank_trades + 1):
        if any(wins_now(g, color) for g in frontier):
            return True
        if depth == bank_trades:
            break
        frontier = [after(g, a) for g in frontier for a in g.playable_actions
                    if a.action_type == ActionType.MARITIME_TRADE]
    return False


def traded(game, proposer, responder, give, want):
    """The position with `give` moved from proposer to responder and `want` the
    other way, as a closed trade would leave it."""
    g = game.copy()
    s = g.state
    p, r = s.color_to_index[proposer], s.color_to_index[responder]
    s.player_state[f"P{p}_{RESOURCES[give]}_IN_HAND"] -= 1
    s.player_state[f"P{r}_{RESOURCES[give]}_IN_HAND"] += 1
    s.player_state[f"P{r}_{RESOURCES[want]}_IN_HAND"] -= 1
    s.player_state[f"P{p}_{RESOURCES[want]}_IN_HAND"] += 1
    g.playable_actions = generate_playable_actions(s)
    return g


def losing_offer(game, color, rng):
    """(give, want, responder) where the trade hands `color` the win, or None."""
    s = game.state
    mine = get_player_freqdeck(s, color)
    others = [c for c in s.colors if c != color]
    found = []
    for want in range(5):
        holders = [c for c in others if get_player_freqdeck(s, c)[want] > 0]
        if not holders:
            continue
        # the card it has most of: the offer a player would actually make
        for give in sorted((i for i in range(5) if i != want and mine[i] > 0),
                           key=lambda i: (-mine[i], i)):
            if wins_now(traded(game, color, holders[0], give, want), color):
                found.append((give, want, holders))
                break
    if not found or wins_without_help(game, color):
        return None
    give, want, holders = found[0]
    return give, want, rng.choice(holders)


def last_decision_end(state, color) -> int:
    """Where the responder's history starts: after the last thing it did."""
    for k in range(len(state.action_records) - 1, -1, -1):
        if state.action_records[k].action.color == color:
            return k + 1
    return 0


def put_on_table(game, talk, proposer, responder, give, want):
    """The offer made, and the responder the one being asked. Anyone polled
    before it would have been asked first; their answers are not what is being
    measured, so the table goes straight to it."""
    g = game.copy()
    t = talk.copy()
    offer = [0] * 10
    offer[give] += 1
    offer[5 + want] += 1
    action = Action(proposer, ActionType.OFFER_TRADE, tuple(offer))
    g.execute(action)
    t.observe(g.state)
    t.record_move(g.state.num_turns, proposer.value, action,
                  PITCH.format(want=RESOURCES[want].lower(), give=RESOURCES[give].lower()))
    g.state.current_player_index = g.state.color_to_index[responder]
    g.playable_actions = generate_playable_actions(g.state)
    assert g.state.current_prompt == ActionPrompt.DECIDE_TRADE
    assert any(a.action_type == ActionType.ACCEPT_TRADE for a in g.playable_actions)
    return g, t


def ready(game) -> bool:
    """On its own turn, after the roll: when an offer can be made."""
    s = game.state
    return (s.current_prompt == ActionPrompt.PLAY_TURN and not s.is_initial_build_phase
            and any(a.action_type == ActionType.END_TURN for a in game.playable_actions))


def find_pair(seed):
    """The first losing offer of a bot game with its control, or None."""
    rng = random.Random(seed)
    game = Game([ValueFunctionPlayer(c) for c in SEATS], seed=seed)
    talk = TableTalk(max_offers_per_turn=MAX_OFFERS)
    talk.observe(game.state)
    earlier = []  # (game, talk) at every ready moment on 5 points or fewer
    while game.winning_color() is None and game.state.num_turns < MAX_TURNS:
        if ready(game):
            s = game.state
            color = s.current_color()
            if vp(s, color) <= FAR_VP:
                earlier.append((game.copy(), talk.copy()))
            elif (vp(s, color) == vp(s, color, actual=True) and not holds_dev_cards(s, color)):
                offer = losing_offer(game, color, rng)
                if offer:
                    give, want, responder = offer
                    control = next(
                        ((g, t) for g, t in reversed(earlier)
                         if g.state.current_color() == color
                         and get_player_freqdeck(g.state, color)[give] > 0
                         and get_player_freqdeck(g.state, responder)[want] > 0),
                        None)
                    if control is None:
                        return None
                    near = put_on_table(game, talk, color, responder, give, want)
                    far = put_on_table(*control, color, responder, give, want)
                    return {"proposer": color.value, "responder": responder.value,
                            "give": RESOURCES[give], "want": RESOURCES[want],
                            "near": {"game": near[0], "talk": near[1],
                                     "turn": near[0].state.num_turns,
                                     "proposer_vp": vp(near[0].state, color)},
                            "far": {"game": far[0], "talk": far[1],
                                    "turn": far[0].state.num_turns,
                                    "proposer_vp": vp(far[0].state, color)}}
        game.play_tick()
        talk.observe(game.state)
    return None


def generate(args):
    started = time.time()
    positions = []
    for seed in range(args.seed_start, args.seed_start + args.games):
        pair = find_pair(seed)
        if pair:
            positions.append({"id": f"seed{seed}", "seed": seed, **pair})
        if (seed - args.seed_start + 1) % 50 == 0:
            print(f"  {seed - args.seed_start + 1}/{args.games} games, "
                  f"{len(positions)} positions", flush=True)
    out = args.out or OUT_DIR / f"loss-positions-{time.strftime('%Y%m%d-%H%M%S')}.pkl"
    OUT_DIR.mkdir(exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump({"games": args.games, "seed_start": args.seed_start,
                     "prompt_version": PROMPT_VERSION, "positions": positions}, fh)
    print(f"{len(positions)} positions from {args.games} games in "
          f"{time.time() - started:.0f}s -> {out}")


def prompt_for(position, condition, persona):
    side = position[condition]
    game, talk = side["game"], side["talk"]
    me = Color[position["responder"]]
    legal = stable_order(game.playable_actions)
    user = build_prompt(game, me, legal, talk, may_offer=False,
                        since=last_decision_end(game.state, me),
                        may_counter=may_counter(game.state, me))
    return ROSTER[persona].system_prompt, user, legal


def ask_one(decider, position, condition, persona):
    system, user, legal = prompt_for(position, condition, persona)
    want = RESOURCES.index(position["want"])
    started = time.time()
    try:
        decision, usage = decider(system, user)
    except DecisionFormatError as exc:
        return {"unusable": str(exc), "usage": exc.usage,
                "seconds": round(time.time() - started, 1)}
    except Exception as exc:  # transport: counted, never fatal to the batch
        return {"error": str(exc), "seconds": round(time.time() - started, 1)}
    base = {"reasoning": decision.reasoning, "thinking": decision.thinking, "say": decision.say, "usage": usage,
            "seconds": round(time.time() - started, 1)}
    if decision.choice == AUTHOR_OFFER:
        # A counter is a rejection to the engine. It still hands over the card
        # if its terms include it and the proposer takes them up.
        gives = [RESOURCES.index(r.upper()) for r in decision.offer_give
                 if r.upper() in RESOURCES]
        return {**base, "answer": "counter", "counter_give": decision.offer_give,
                "counter_want": decision.offer_want, "counter_hands_it": want in gives}
    if isinstance(decision.choice, int) and 0 <= decision.choice < len(legal):
        t = legal[decision.choice].action_type
        return {**base, "answer": "accept" if t == ActionType.ACCEPT_TRADE else "reject"}
    return {"unusable": f"index {decision.choice} out of range", "usage": usage,
            "seconds": base["seconds"]}


def ask(args):
    with open(args.positions, "rb") as fh:
        bundle = pickle.load(fh)
    if bundle["prompt_version"] != PROMPT_VERSION:
        sys.exit(f"positions were made under prompt v{bundle['prompt_version']}, "
                 f"the prompt is now v{PROMPT_VERSION}")
    positions = bundle["positions"][:args.limit] if args.limit else bundle["positions"]
    personas = args.personas.split(",")
    models = args.models.split(",")
    jobs = [(m, p, c, pe, k) for p in positions for c in ("near", "far") for pe in personas
            for m in models for k in range(args.samples)]
    if args.measure:
        jobs = [(m, positions[0], c, personas[0], 0) for m in models for c in ("near", "far")]
    random.Random(0).shuffle(jobs)

    if not args.dry_run:
        load_env()
    budget = {"max_tokens": args.max_tokens, "timeout": 300.0} if args.max_tokens else {}
    deciders = {m: ScriptedDecider(heuristic_script("trader", rng=random.Random(1)),
                                   label="dry-run")
                # With one sample no prompt is ever sent twice, so a cache write
                # is a 1.25x fee for nothing (DESIGN.md, *Prompt caching*).
                if args.dry_run else make_decider(m, effort=args.effort, flex=args.flex,
                                                  cache_writes=args.samples > 1, **budget)
                for m in models}
    tag = "dryrun-" if args.dry_run else ("measure-" if args.measure else "")
    out = args.out or OUT_DIR / f"loss-probe-{tag}{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    lock = threading.Lock()
    done = []
    with out.open("a", buffering=1) as fh:
        fh.write(json.dumps({"kind": "plan", "positions": str(args.positions),
                             "models": models, "personas": personas,
                             "samples": args.samples, "jobs": len(jobs),
                             "effort": args.effort, "flex": args.flex,
                             "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
        print(f"{len(jobs)} calls -> {out}", flush=True)

        def run(job):
            model, position, condition, persona, k = job
            answer = ask_one(deciders[model], position, condition, persona)
            row = {"kind": "answer", "model": model, "position": position["id"],
                   "condition": condition, "persona": persona, "sample": k,
                   "answer": answer,
                   "usd": None if args.dry_run else usd(answer.get("usage"), model),
                   "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            with lock:
                fh.write(json.dumps(row, default=str) + "\n")
                done.append(row)
                n = len(done)
            flag = ("UNUSABLE " + answer["unusable"] if "unusable" in answer
                    else "ERROR " + answer["error"] if "error" in answer else "")
            if flag or n % 50 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} {flag}".rstrip(), flush=True)

        with ThreadPoolExecutor(max_workers=1 if args.dry_run else args.workers) as pool:
            for f in as_completed([pool.submit(run, j) for j in jobs]):
                f.result()
    spent = sum(r["usd"] or 0 for r in done)
    print(f"{len(done)} answers, ${spent:.4f} -> {out}")


# ------------------------------------------------------------------- reading it

#: Keyword reading of the reasoning, crude and reported as such.
MENTIONS_SCORE = re.compile(r"\b(\d+ ?vp|victory points?|points?|lead(s|ing|er)?|"
                            r"win|wins|winning|close to|near(ly)? (the )?(end|win))\b", re.I)


def hands_it(answer) -> bool:
    return answer["answer"] == "accept" or (answer["answer"] == "counter"
                                            and answer["counter_hands_it"])


def report(args):
    rows = []
    for path in args.answers:
        rows += [r for r in map(json.loads, path.open()) if r["kind"] == "answer"]
    usable = [r for r in rows if "answer" in r["answer"]]
    print(f"{len(usable)}/{len(rows)} usable answers")
    groups = defaultdict(list)
    for r in usable:
        groups[(r["model"], r["persona"])].append(r)
    for (model, persona), group in sorted(groups.items()):
        print(f"\n{model}, {persona}: ${sum(price(r) for r in rows if r['model'] == model and r['persona'] == persona):.3f}")
        rate = {}
        for cond in ("near", "far"):
            g = [r for r in group if r["condition"] == cond]
            if not g:
                continue
            n = len(g)
            acc = sum(r["answer"]["answer"] == "accept" for r in g)
            cnt = sum(r["answer"]["answer"] == "counter" for r in g)
            cnt_hand = sum(r["answer"]["answer"] == "counter" and r["answer"]["counter_hands_it"]
                           for r in g)
            score = sum(bool(MENTIONS_SCORE.search(r["answer"]["reasoning"] or "")) for r in g)
            rate[cond] = {}
            for r in g:
                rate[cond].setdefault(r["position"], []).append(r["answer"]["answer"] == "accept")
            print(f"  {cond:4s}  accept {acc}/{n} ({acc / n:.0%})   counter {cnt} "
                  f"(of which offer the card {cnt_hand})   reasoning mentions the score {score}")
        if {"near", "far"} <= set(rate):
            shared = sorted(set(rate["near"]) & set(rate["far"]))
            d = [statistics.mean(rate["near"][p]) - statistics.mean(rate["far"][p])
                 for p in shared]
            up, down = sum(x > 0 for x in d), sum(x < 0 for x in d)
            k = up + down
            sign_p = (min(1.0, 2 * sum(math.comb(k, i) for i in range(max(up, down), k + 1))
                          / 2 ** k) if k else 1.0)
            rng = random.Random(0)
            boot = sorted(statistics.mean(rng.choice(d) for _ in d) for _ in range(5000))
            print(f"  near - far, paired on {len(shared)} positions: {statistics.mean(d):+.0%} "
                  f"(95% {boot[124]:+.0%} to {boot[4874]:+.0%}), "
                  f"more/less/same {up}/{down}/{len(d) - k}, sign test p={sign_p:.2g}")
    spent = sum(price(r) for r in rows)
    print(f"\ncost ${spent:.3f}")


def price(row) -> float:
    """What a row cost, priced again from its usage. Rows written before the
    request time was recorded fall back to the time the answer came back, so an
    off-peak DeepSeek run is not reported at the peak rate it never paid."""
    if row.get("usd") is None:
        return 0.0
    usage = dict(row["answer"].get("usage") or {})
    if not usage.get("requested_at") and row.get("at"):
        usage["requested_at"] = datetime.strptime(row["at"], "%Y-%m-%dT%H:%M:%S%z").timestamp()
    return usd(usage, row["model"]) or 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--games", type=int, default=400)
    g.add_argument("--seed-start", type=int, default=20000)
    g.add_argument("--out", type=Path)
    a = sub.add_parser("ask")
    a.add_argument("positions", type=Path)
    a.add_argument("--models", default="gpt-5.6-luna")
    a.add_argument("--personas", default="Plain,Cooperator,Quiet builder,Hard bargainer")
    # One answer per position: on the 2026-09-23 batch, 90 positions asked once
    # gave nearly the interval of 90 asked three times. More positions, which
    # cost nothing to make, buy precision; repeats mostly buy cost.
    a.add_argument("--samples", type=int, default=1)
    a.add_argument("--limit", type=int, default=None, help="use only the first N positions")
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--dry-run", action="store_true", help="scripted answers, spends nothing")
    a.add_argument("--measure", action="store_true", help="two calls per model, then stop")
    a.add_argument("--max-tokens", type=int, default=None)
    a.add_argument("--effort", default=None)
    a.add_argument("--flex", action="store_true",
                   help="OpenAI's half-price tier: slower, and waited out when busy")
    a.add_argument("--out", type=Path)
    r = sub.add_parser("report")
    r.add_argument("answers", nargs="+", type=Path)
    args = ap.parse_args(argv)
    {"generate": generate, "ask": ask, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()

"""Does a model find the road that wins the game?

Recorded matches give too few such positions (10 in nine matches) and give
them all on the same few boards. This fabricates them instead: four value bots
play each other on fresh seeds, and the first position of each game where
building a road — and no other kind of move — would win on the spot is kept,
or where it would if the player held one more wood and one more brick. A win is
the one position where the right move is not a matter of judgement, so the
answer is scored as right or wrong without arguing about strategy.

The agent's prompt is built by `arena/prompt.py` for the player to move, as in
a match, minus what bots never produce: table talk and the agent's own note.

    .venv/bin/python scripts/win_probe.py generate --games 600
    .venv/bin/python scripts/win_probe.py ask experiments/win-positions-<ts>.pkl --dry-run
    .venv/bin/python scripts/win_probe.py ask experiments/win-positions-<ts>.pkl --measure
    .venv/bin/python scripts/win_probe.py ask experiments/win-positions-<ts>.pkl

Nothing here reads or writes runs/.
"""

import argparse
import itertools
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
from catanatron.models.enums import ActionType  # noqa: E402
from catanatron.models.player import Color  # noqa: E402
from catanatron.players.value import ValueFunctionPlayer  # noqa: E402

from arena.deciders import DecisionFormatError, ScriptedDecider, make_decider  # noqa: E402
from arena.env import load_env  # noqa: E402
from arena.llm_player import may_offer, stable_order  # noqa: E402
from arena.personas import PLAIN  # noqa: E402
from arena.play import usd  # noqa: E402
from arena.prompt import AUTHOR_OFFER, PROMPT_VERSION, build_prompt, describe_action  # noqa: E402
from arena.scripted import heuristic_script  # noqa: E402
from arena.table_talk import TableTalk  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "experiments"
SEATS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
#: The table the v3 matches play at.
MAX_OFFERS = 2
#: Bots occasionally stall; a game this long is abandoned, not waited on.
MAX_TURNS = 400


def wins(game, action, color) -> bool:
    after = game.copy()
    after.execute(action, validate_action=False)
    return after.winning_color() == color


def road_only_win(game, color) -> bool:
    """Some road wins and no other kind of move does, so finding a winning road
    is the whole test. Often more than one does: both ends of the same chain."""
    winning = [a for a in game.playable_actions if wins(game, a, color)]
    return bool(winning) and all(a.action_type == ActionType.BUILD_ROAD for a in winning)


def with_gift(game, color):
    """The same position with one more wood and one more brick in hand. The bank
    gives them up, so its counts stay consistent with the hands; None when it
    has none to give."""
    if game.state.resource_freqdeck[0] < 1 or game.state.resource_freqdeck[1] < 1:
        return None
    g = game.copy()
    s = g.state
    i = s.color_to_index[color]
    s.player_state[f"P{i}_WOOD_IN_HAND"] += 1
    s.player_state[f"P{i}_BRICK_IN_HAND"] += 1
    s.resource_freqdeck[0] -= 1
    s.resource_freqdeck[1] -= 1
    g.playable_actions = generate_playable_actions(s)
    return g


def previous_turn_end(state, color) -> int:
    """Where this seat's history starts: after the last thing it did on its
    previous turn, which is where an agent would last have been asked."""
    records = state.action_records
    k = len(records)
    while k > 0 and records[k - 1].action.color == color:
        k -= 1  # this turn's own moves so far
    while k > 0 and records[k - 1].action.color != color:
        k -= 1
    return k


def find_position(seed):
    """The first road-only win of a bot game, or None. One per game, so no two
    positions share a board.

    The seed fixes the board, not the game: the value bots break ties in an
    order that changes between runs, so the same seed plays out differently each
    time. The pickled state is the record of a position, never the seed."""
    game = Game([ValueFunctionPlayer(c) for c in SEATS], seed=seed)
    while game.winning_color() is None and game.state.num_turns < MAX_TURNS:
        s = game.state
        if s.current_prompt.value == "PLAY_TURN" and not s.is_initial_build_phase:
            color = s.current_color()
            if s.player_state[f"P{s.color_to_index[color]}_ACTUAL_VICTORY_POINTS"] >= 7:
                if road_only_win(game, color):
                    return game.copy(), color, False
                gifted = with_gift(game, color)
                if gifted and road_only_win(gifted, color):
                    return gifted, color, True
        game.play_tick()
    return None


def generate(args):
    started = time.time()
    positions = []
    for seed in range(args.seed_start, args.seed_start + args.games):
        found = find_position(seed)
        if found:
            game, color, gift = found
            positions.append({"id": f"seed{seed}", "seed": seed, "color": color.value,
                              "gift": gift, "turn": game.state.num_turns, "game": game})
        if (seed - args.seed_start + 1) % 100 == 0:
            print(f"  {seed - args.seed_start + 1}/{args.games} games, {len(positions)} positions",
                  flush=True)
    out = args.out or OUT_DIR / f"win-positions-{time.strftime('%Y%m%d-%H%M%S')}.pkl"
    OUT_DIR.mkdir(exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump({"games": args.games, "seed_start": args.seed_start,
                     "prompt_version": PROMPT_VERSION, "positions": positions}, fh)
    print(f"{len(positions)} positions from {args.games} games "
          f"({sum(p['gift'] for p in positions)} with the gift) in "
          f"{time.time() - started:.0f}s -> {out}")


def prompt_for(position):
    game = position["game"]
    color = Color[position["color"]]
    legal = stable_order(game.playable_actions)
    talk = TableTalk(max_offers_per_turn=MAX_OFFERS)
    user = build_prompt(game, color, legal, talk, may_offer(game.state, legal, talk, color),
                        since=previous_turn_end(game.state, color))
    return PLAIN.system_prompt, user, legal


def cost(usage, model):
    return usd(usage, model)


def ask_one(decider, position, model):
    system, user, legal = prompt_for(position)
    color = Color[position["color"]]
    started = time.time()
    try:
        decision, usage = decider(system, user)
    except DecisionFormatError as exc:
        return {"unusable": str(exc), "usage": exc.usage, "seconds": round(time.time() - started, 1)}
    except Exception as exc:  # transport: counted, never fatal to the batch
        return {"error": str(exc), "seconds": round(time.time() - started, 1)}
    if decision.choice == AUTHOR_OFFER:
        action, described = None, f"offer {decision.offer_give} for {decision.offer_want}"
    elif isinstance(decision.choice, int) and 0 <= decision.choice < len(legal):
        action = legal[decision.choice]
        described = describe_action(action)
    else:
        return {"unusable": f"index {decision.choice} out of range", "usage": usage,
                "seconds": round(time.time() - started, 1)}
    return {"choice": decision.choice, "described": described, "reasoning": decision.reasoning, "thinking": decision.thinking,
            "say": decision.say, "usage": usage, "seconds": round(time.time() - started, 1),
            "takes_win": action is not None and wins(position["game"], action, color),
            "builds_road": action is not None and action.action_type == ActionType.BUILD_ROAD}


def ask(args):
    with open(args.positions, "rb") as fh:
        bundle = pickle.load(fh)
    positions = bundle["positions"][:args.limit] if args.limit else bundle["positions"]
    if args.only:
        wanted = args.only.split(",")
        positions = [p for p in bundle["positions"] if p["id"] in wanted]
    if bundle["prompt_version"] != PROMPT_VERSION:
        sys.exit(f"positions were found under prompt v{bundle['prompt_version']}, "
                 f"the prompt is now v{PROMPT_VERSION}")
    models = args.models.split(",")
    jobs = [(m, p, k) for p in positions for m in models for k in range(args.samples)]
    if args.measure:
        jobs = [(m, positions[0], 0) for m in models]
    random.Random(0).shuffle(jobs)  # spread each model's calls over the whole run

    if not args.dry_run:
        load_env()
    # A thinking model spends its output budget reasoning; room for it must be
    # asked for, or the reply comes back cut off (see arena/deciders.py).
    budget = {"max_tokens": args.max_tokens, "timeout": 300.0} if args.max_tokens else {}
    if args.no_thinking and not all(m.startswith("deepseek") for m in models):
        sys.exit("--no-thinking is DeepSeek's switch; other providers reject the field")
    if args.no_thinking:
        budget["extra_body"] = {"thinking": {"type": "disabled"}}
    setting = {"effort": args.effort, "thinking": not args.no_thinking, "flex": args.flex}
    deciders = {m: ScriptedDecider(heuristic_script("trader", rng=random.Random(1)), label="dry-run")
                # With one sample no prompt is sent twice, so a cache write is a
                # 1.25x fee for nothing (DESIGN.md, *Prompt caching*).
                if args.dry_run else make_decider(m, effort=args.effort, flex=args.flex,
                                                  cache_writes=args.samples > 1, **budget)
                for m in models}
    tag = "dryrun-" if args.dry_run else ("measure-" if args.measure else "")
    out = args.out or OUT_DIR / f"win-probe-{tag}{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    lock = threading.Lock()
    done = []
    with out.open("a", buffering=1) as fh:
        fh.write(json.dumps({"kind": "plan", "positions": str(args.positions), "models": models,
                             "samples": args.samples, "jobs": len(jobs), **setting,
                             "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
        print(f"{len(jobs)} calls -> {out}", flush=True)

        def run(job):
            model, position, k = job
            answer = ask_one(deciders[model], position, model)
            row = {"kind": "answer", "model": model, "position": position["id"],
                   "gift": position["gift"], "sample": k, **setting, "answer": answer,
                   "usd": None if args.dry_run else cost(answer.get("usage"), model),
                   "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            with lock:
                fh.write(json.dumps(row, default=str) + "\n")
                done.append(row)
                n = len(done)
            flag = ("UNUSABLE " + answer["unusable"] if "unusable" in answer
                    else "ERROR " + answer["error"] if "error" in answer else "")
            print(f"  {n}/{len(jobs)} {model} {flag}".rstrip(), flush=True)

        with ThreadPoolExecutor(max_workers=1 if args.dry_run else args.workers) as pool:
            for f in as_completed([pool.submit(run, j) for j in jobs]):
                f.result()
    for m in models:
        rows = [r for r in done if r["model"] == m]
        ok = [r for r in rows if "choice" in r["answer"]]
        wins_n = sum(r["answer"]["takes_win"] for r in ok)
        usd = sum(r["usd"] or 0 for r in rows)
        print(f"{m:16s} {len(ok)}/{len(rows)} usable, took the win {wins_n}/{len(ok)}, ${usd:.4f} at peak")
    print(f"answers: {out}")


# ------------------------------------------------------------------- reading it

#: Keyword reading of the reasoning. Crude on purpose and reported as such: the
#: split it shows (most of one model's wins say so, few of another's) is far
#: wider than any phrasing it can misread.
SAYS_WIN = re.compile(r"\b(win|wins|winning|won|game[- ]ending|end the game|reach(es|ing)? 10|"
                      r"to 10|10 (vp|victory|points))\b", re.I)


def settle(position, answer):
    """What an answer did with the win: 'won', or it played Road Building with a
    winning road still to place ('road_building'), or it did something else with
    the win still one road away ('deferred'), or it gave the win up ('gone').
    Road Building is a winning line, not a miss: the two free roads include the
    winning one, and the models that play it say so."""
    if answer["takes_win"]:
        return "won"
    color = Color[position["color"]]
    if answer["choice"] == AUTHOR_OFFER:
        return "deferred"  # an offer moves no cards
    _, _, legal = prompt_for(position)
    move = legal[answer["choice"]]
    after = position["game"].copy()
    after.execute(move, validate_action=False)
    still = after.state.current_color() == color and any(
        a.action_type == ActionType.BUILD_ROAD and wins(after, a, color)
        for a in after.playable_actions)
    if move.action_type == ActionType.PLAY_ROAD_BUILDING and still:
        return "road_building"
    return "deferred" if still else "gone"


def setting_of(row):
    tail = []
    if row.get("effort"):
        tail.append(f"effort {row['effort']}")
    if row.get("thinking") is False:
        tail.append("no thinking")
    return row["model"] + (f" ({', '.join(tail)})" if tail else "")


def billed(row) -> float:
    """What an answer cost, priced from its usage: the flex tier when the reply
    reported it, DeepSeek's off-peak rate when the request was made off-peak.
    Rows from before the request time was recorded fall back to the time the
    answer came back."""
    usage = dict(row["answer"].get("usage") or {})
    if not usage:
        return 0.0
    if not usage.get("requested_at"):
        usage["requested_at"] = datetime.strptime(row["at"], "%Y-%m-%dT%H:%M:%S%z").timestamp()
    return cost(usage, row["model"]) or 0.0


def report(args):
    rows, walls, bundles = [], [], {}
    for path in args.answers:
        lines = [json.loads(l) for l in path.open()]
        plan = next(l for l in lines if l["kind"] == "plan")
        answers = [l for l in lines if l["kind"] == "answer"]
        if plan["positions"] not in bundles:
            with open(plan["positions"], "rb") as fh:
                bundles[plan["positions"]] = {p["id"]: p for p in pickle.load(fh)["positions"]}
        for a in answers:
            a["_position"] = bundles[plan["positions"]][a["position"]]
        rows += answers
        if answers:
            end = max(datetime.strptime(a["at"], "%Y-%m-%dT%H:%M:%S%z") for a in answers)
            start = datetime.strptime(plan["started"], "%Y-%m-%dT%H:%M:%S%z")
            walls.append((path.name, sorted({setting_of(a) for a in answers}),
                          (end - start).total_seconds()))

    per = defaultdict(lambda: defaultdict(list))
    groups = defaultdict(list)
    for r in rows:
        groups[setting_of(r)].append(r)
    for name, group in groups.items():
        usable = [r for r in group if "choice" in r["answer"]]
        fate = {id(r): settle(r["_position"], r["answer"]) for r in usable}
        line = [r for r in usable if fate[id(r)] in ("won", "road_building")]
        for r in usable:
            per[name][r["position"]].append(fate[id(r)] in ("won", "road_building"))
        usage = [r["answer"]["usage"] for r in usable]
        secs = [r["answer"]["seconds"] for r in usable]
        spent = sum(billed(r) for r in usable)
        n = len(usable)
        print(f"\n{name}: {n}/{len(group)} usable answers on {len(per[name])} positions")
        print(f"  winning line       {len(line)} ({len(line) / n:.0%})  "
              f"[built the road {sum(f == 'won' for f in fate.values())}, "
              f"Road Building {sum(f == 'road_building' for f in fate.values())}]")
        print(f"  said it wins       {sum(bool(SAYS_WIN.search(r['answer']['reasoning'] or '')) for r in line)}")
        print(f"  deferred / gone    {sum(f == 'deferred' for f in fate.values())} / "
              f"{sum(f == 'gone' for f in fate.values())}")
        print(f"  output tokens      mean {statistics.mean(u['output_tokens'] for u in usage):,.0f}, "
              f"max {max(u['output_tokens'] for u in usage):,}, total {sum(u['output_tokens'] for u in usage):,}")
        print(f"  seconds per call   mean {statistics.mean(secs):.1f}, median {statistics.median(secs):.1f}, "
              f"all calls end to end {sum(secs) / 60:.0f} min")
        print(f"  cost               ${spent:.3f} as billed (flex and off-peak applied per request)")

    names = sorted(per, key=lambda k: -sum(sum(v) for v in per[k].values()))
    for a, b in itertools.combinations(names, 2):
        shared = sorted(set(per[a]) & set(per[b]))
        if len(shared) < 2:
            continue
        d = [statistics.mean(per[a][p]) - statistics.mean(per[b][p]) for p in shared]
        up, down = sum(x > 0 for x in d), sum(x < 0 for x in d)
        k = up + down
        sign_p = min(1.0, 2 * sum(math.comb(k, i) for i in range(max(up, down), k + 1)) / 2 ** k) if k else 1.0
        rng = random.Random(0)
        boot = sorted(statistics.mean(rng.choice(d) for _ in d) for _ in range(5000))
        print(f"\n{a} vs {b}, {len(shared)} shared positions: gap {statistics.mean(d):+.0%} "
              f"(95% {boot[124]:+.0%} to {boot[4874]:+.0%}), better/worse/tie {up}/{down}/{len(d) - k}, "
              f"sign test p={sign_p:.1g}")
    for name, who, secs in walls:
        print(f"\nwall clock {name}: {secs / 60:.1f} min for {', '.join(who)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--games", type=int, default=600)
    g.add_argument("--seed-start", type=int, default=9000)
    g.add_argument("--out", type=Path)
    a = sub.add_parser("ask")
    a.add_argument("positions", type=Path)
    a.add_argument("--models", default="deepseek-flash,gpt-5.6-luna")
    a.add_argument("--samples", type=int, default=3)
    a.add_argument("--limit", type=int, default=None, help="use only the first N positions")
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--dry-run", action="store_true", help="scripted answers, spends nothing")
    a.add_argument("--measure", action="store_true", help="one call per model, then stop")
    a.add_argument("--max-tokens", type=int, default=None,
                   help="output budget per call, reasoning included; the provider default otherwise")
    a.add_argument("--only", default=None, help="comma-separated position ids, e.g. seed30322")
    a.add_argument("--effort", default=None, help="reasoning effort; the provider default otherwise")
    a.add_argument("--no-thinking", action="store_true",
                   help="turn DeepSeek's thinking mode off (its default is on, at effort high)")
    a.add_argument("--flex", action="store_true",
                   help="OpenAI's half-price tier: slower, and waited out when busy")
    a.add_argument("--out", type=Path)
    r = sub.add_parser("report")
    r.add_argument("answers", nargs="+", type=Path)
    args = ap.parse_args(argv)
    {"generate": generate, "ask": ask, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()

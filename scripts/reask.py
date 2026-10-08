"""Ask a model again about positions it already played, under different prompts.

A whole match to test a prompt change costs ~$0.25 and twenty minutes, and says
little: one game, one board, dice. The decisions that went wrong are already on
disk, and `arena/rebuild.py` puts each one back exactly — board, hand, table
talk, offer budget — so the same question can be asked again, many times, for a
fraction of a cent each.

Six sets of positions, all chosen by the model the first time:

  missed_win  a win was available this turn and the move played gave it up
  dev_buy     bought a development card where the bot would not have
  control     built what the bot would also have built — a change that makes
              these worse is breaking something
  road_title  some legal road takes longest road, whatever was played; the
              engine knows which, the prompt leaves the agent to work it out
              from a list of edges
  leader_offer  answering an offer from the player leading on public points
  offer_control answering an offer from anyone else, as many as leader_offer:
                a change that makes the agent refuse everyone is not the same
                as one that makes it wary of the leader
  robber      moving the robber while one opponent leads the others on public
              points and some legal move steals from it

`--model` asks a different model than the one that played, on the same
positions: how another model would have handled them, without a match.

Conditions, one per prompt version (PROMPT_VERSION in arena/prompt.py):

  as_played     whichever version the run was played under — the control
  v1            the first four matches: before the fixes to the VP line and
                the road length
  v2            the 2026-09-18 overnight batch: those fixes, but no roads,
                ports, history, bank, own note, discards or counter-offers
  current       the prompt as it is now
  current_high  the prompt as it is now, with reasoning effort 'high'
  win_rule      the prompt as it is now, with the win condition stated in the
                rules. RULES never says how the game is won; the agents aim the
                robber at the leader but trade with it as readily as with anyone
  plain_high    the prompt as it is now, seated as Plain, effort 'high'
  saboteur_high the same, seated as the Saboteur. Whoever played the position,
                these two differ in the persona alone. Together they are the
                manipulation check for any match with a saboteur in it: if the
                model plays both alike, a match would compare two identical
                tables

The control exists because a model does not answer the same prompt the same
way twice; without it, any change would be credited to the fix.

    .venv/bin/python scripts/reask.py --dry-run runs/<match>.jsonl ...   # free
    .venv/bin/python scripts/reask.py --measure runs/<match>.jsonl ...   # 3 calls
    .venv/bin/python scripts/reask.py runs/<match>.jsonl ...             # the batch

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

from catanatron.models.enums import ActionType  # noqa: E402
from catanatron.players.value import ValueFunctionPlayer  # noqa: E402
from catanatron.state_functions import (  # noqa: E402
    get_longest_road_color, get_visible_victory_points)

from arena.deciders import DecisionFormatError, ScriptedDecider, make_decider  # noqa: E402
from arena.env import load_env  # noqa: E402
from arena.personas import ROSTER, RULES  # noqa: E402
from arena.play import usd  # noqa: E402
from arena.prompt import AUTHOR_OFFER, PROMPT_VERSION, describe_action  # noqa: E402
from arena.rebuild import detach, plies  # noqa: E402
from arena.scripted import heuristic_script  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "experiments"

CONDITIONS = {
    "as_played": {"prompt": "as_played", "effort": None},
    "v1": {"prompt": "v1", "effort": None},
    "v2": {"prompt": "v2", "effort": None},
    "current": {"prompt": "current", "effort": None},
    "current_high": {"prompt": "current", "effort": "high"},
    "win_rule": {"prompt": "current", "effort": None, "rules":
                 "The first player to reach 10 victory points wins the game at once, "
                 "and everyone else loses."},
    # 'high' reasons past the default 4,096-token cap: 5 of 166 win-probe
    # answers were cut off there (2026-10-07).
    "plain_high": {"prompt": "current", "effort": "high", "persona": "Plain",
                   "budget": {"max_tokens": 16384, "timeout": 300.0}},
    "saboteur_high": {"prompt": "current", "effort": "high", "persona": "Saboteur",
                      "budget": {"max_tokens": 16384, "timeout": 300.0}},
}

#: Moves that can complete a win inside one turn without the dice or another
#: player. Development cards are left out: what one draws is not known yet.
WINNING_MOVES = (
    ActionType.MARITIME_TRADE, ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT,
    ActionType.BUILD_ROAD, ActionType.PLAY_YEAR_OF_PLENTY, ActionType.PLAY_MONOPOLY,
    ActionType.PLAY_ROAD_BUILDING,
)
WIN_DEPTH = 3


def v2(user: str) -> str:
    """The prompt before version 3 (arena/prompt.py, PROMPT_VERSION): no roads,
    ports, bank, history, own note, discards or counter-offers. Only meaningful
    on positions from a v2 or v1 run — a v3 run's table talk can hold counters
    and directed offers, which no earlier prompt ever showed."""
    user = re.sub(r"\n\nROADS \(.*?(?=\n\nPLAYERS:)", "", user, flags=re.S)
    user = re.sub(r"\n\nBANK:.*?(?=\n\nTABLE TALK)", "", user, flags=re.S)
    user = re.sub(r"\n\nYOUR OWN NOTE [^\n]*\n[^\n]*", "", user)
    user = re.sub(r"\n\nDISCARD: [^\n]*", "", user)
    user = re.sub(r"\n  \[\d+\] take up [^\n]*", "", user)
    user = user.replace(", and taking up a counter-offer uses one.", ".")
    user = re.sub(r"\n\nYou may instead counter [^\n]*", "", user)
    return re.sub(r"\| ports: ([^\n]*)", _v2_ports, user)


def _v2_ports(m) -> str:
    """'2:1 wood, 3:1' back to 'WOOD' and '3:1', in the old string sort order."""
    ports = [p[4:].upper() if p.startswith("2:1 ") else p for p in m.group(1).split(", ")]
    return f"| ports: {', '.join(sorted(ports))}"


def v1(user: str) -> str:
    """`v2`, and before that the two fixes to the players section: no split of
    the agent's own VP total, no road length or knights played."""
    user = re.sub(r" \(\d+ that everyone can see \+ \d+ from victory point cards "
                  r"only you can see\)", "", v2(user))
    return re.sub(r" \(longest continuous road \d+, \d+ knights played\)", "", user)


PROMPT_VERSIONS = {"v1": v1, "v2": v2, "current": lambda user: user}


def played_version(start: dict) -> str:
    """The prompt version a run was played under. From version 3 the run says so;
    before that the batch boundary is by time: the first four real matches
    were v1, the 2026-09-18 02:18 batch v2."""
    v = start.get("prompt_version")
    if v is not None:
        return "current" if v == PROMPT_VERSION else f"v{v}"
    return "v1" if start["game_id"] < "20260918-0215" else "v2"


def win_within(game, color, depth) -> bool:
    if game.winning_color() == color:
        return True
    if depth == 0:
        return False
    for action in game.playable_actions:
        if action.action_type in WINNING_MOVES:
            after = game.copy()
            after.execute(action, validate_action=False)
            if win_within(after, color, depth - 1):
                return True
    return False


def find_positions(paths, controls=20, seed=0):
    """Every position in the four sets, each detached so it can be played on."""
    found = {"missed_win": [], "dev_buy": [], "control": [], "road_title": [],
             "leader_offer": [], "offer_control": [], "robber": []}
    control_pool, offer_pool = [], []
    for path in paths:
        for ply in plies(path):
            state = ply.game.state
            if ply.chosen_by == "decision" and state.current_prompt.value == "DECIDE_TRADE":
                bot = ValueFunctionPlayer(ply.action.color).decide(ply.game, ply.legal())
                (found["leader_offer"] if offer_from_leader(state) else offer_pool).append(
                    (detach(ply), bot))
                continue
            if ply.chosen_by == "decision" and state.current_prompt.value == "MOVE_ROBBER":
                target = leading_opponent(state, ply.action.color)
                if target and any(a.value[1] == target for a in ply.legal()):
                    bot = ValueFunctionPlayer(ply.action.color).decide(ply.game, ply.legal())
                    found["robber"].append((detach(ply), bot))
                continue
            if ply.chosen_by != "decision" or state.current_prompt.value != "PLAY_TURN":
                continue
            color, game = ply.action.color, ply.game
            bot = ValueFunctionPlayer(color).decide(game, ply.legal())
            kind = ply.action.action_type
            me = state.colors.index(color)
            if (state.player_state[f"P{me}_ACTUAL_VICTORY_POINTS"] >= 7
                    and win_within(game, color, WIN_DEPTH)
                    and not takes_win(ply.game, color, ply.action)):
                found["missed_win"].append((detach(ply), bot))
            if any(takes_title(game, color, a) for a in ply.legal()):
                found["road_title"].append((detach(ply), bot))
            if kind == ActionType.BUY_DEVELOPMENT_CARD and bot.action_type != kind:
                found["dev_buy"].append((detach(ply), bot))
            if bot == ply.action and kind in (ActionType.BUILD_SETTLEMENT,
                                              ActionType.BUILD_CITY, ActionType.BUILD_ROAD):
                control_pool.append((detach(ply), bot))
    found["control"] = random.Random(seed).sample(control_pool, min(controls, len(control_pool)))
    found["offer_control"] = random.Random(seed).sample(
        offer_pool, min(len(found["leader_offer"]), len(offer_pool)))
    return found


def offer_from_leader(state) -> bool:
    """Whether the offer on the table comes from a player strictly ahead of every
    other seat on public points, the score the agent is shown in PLAYERS. The
    engine's own total counts hidden victory point cards nobody else can see."""
    proposer = state.colors[state.current_trade[10]]
    vp = {c: get_visible_victory_points(state, c) for c in state.colors}
    return all(vp[proposer] > vp[c] for c in state.colors if c != proposer)


def leading_opponent(state, me):
    """The opponent strictly ahead of every other opponent on public points, or
    None on a tie. The agent's own score is left out: a saboteur in the lead
    still has someone to stop."""
    vp = {c: get_visible_victory_points(state, c) for c in state.colors if c != me}
    top = max(vp.values())
    leaders = [c for c, v in vp.items() if v == top]
    return leaders[0] if len(leaders) == 1 else None


def robs_leader(state, me, action) -> bool:
    return (action is not None and action.action_type == ActionType.MOVE_ROBBER
            and action.value[1] is not None
            and action.value[1] == leading_opponent(state, me))


def takes_win(game, color, action) -> bool:
    """Whether a move keeps the win in reach this turn. An offer does not: it
    spends the move on another player's answer."""
    if action is None or action.action_type == ActionType.OFFER_TRADE:
        return False
    after = game.copy()
    after.execute(action, validate_action=False)
    return win_within(after, color, WIN_DEPTH - 1)


def takes_title(game, color, action) -> bool:
    """Whether a road moves longest road to `color`. Holding it already does
    not count: there is nothing to take."""
    if action is None or action.action_type != ActionType.BUILD_ROAD:
        return False
    if get_longest_road_color(game.state) == color:
        return False
    after = game.copy()
    after.execute(action, validate_action=False)
    return get_longest_road_color(after.state) == color


def ask(decider, ply, bot, which, condition):
    system, user = ply.prompt()
    version = CONDITIONS[condition]["prompt"]
    if version == "as_played":
        version = played_version(ply.start)
    user = PROMPT_VERSIONS[version](user)
    persona = CONDITIONS[condition].get("persona")
    if persona:
        system = ROSTER[persona].system_prompt
    rule = CONDITIONS[condition].get("rules")
    if rule:
        # among the rules, before the persona's style, as if it had always been there
        assert RULES in system
        system = system.replace(RULES, f"{RULES}\n\n{rule}")
    legal = ply.legal()
    started = time.time()
    try:
        decision, usage = decider(system, user)
    except DecisionFormatError as exc:
        return {"unusable": str(exc), "usage": exc.usage, "seconds": round(time.time() - started, 1)}
    except Exception as exc:  # transport: counted, never fatal to the batch
        return {"error": str(exc), "seconds": round(time.time() - started, 1)}
    color = ply.action.color
    if decision.choice == AUTHOR_OFFER:
        action, described = None, f"offer {decision.offer_give} for {decision.offer_want}"
    elif isinstance(decision.choice, int) and 0 <= decision.choice < len(legal):
        action = legal[decision.choice]
        described = describe_action(action)
    else:
        return {"unusable": f"index {decision.choice} out of range", "usage": usage}
    out = {
        "choice": decision.choice, "described": described,
        "reasoning": decision.reasoning, "thinking": decision.thinking, "say": decision.say, "usage": usage,
        "seconds": round(time.time() - started, 1),
        "same_as_played": action == ply.action,
        "agrees_with_bot": action == bot,
        "buys_dev": action is not None and action.action_type == ActionType.BUY_DEVELOPMENT_CARD,
        "accepts": action is not None and action.action_type == ActionType.ACCEPT_TRADE,
        "counters": decision.choice == AUTHOR_OFFER,
    }
    if which == "missed_win":
        out["takes_win"] = takes_win(ply.game, color, action)
    if which == "road_title":
        out["takes_title"] = takes_title(ply.game, color, action)
    if which == "robber":
        out["robs_leader"] = robs_leader(ply.game.state, color, action)
    return out


def cost(usage, model):
    return usd(usage, model)


def summarize(rows):
    table = {}
    for r in rows:
        if r.get("kind") != "answer":
            continue
        key = (r["set"], r["condition"])
        t = table.setdefault(key, {"n": 0, "unusable": 0, "takes_win": 0, "buys_dev": 0,
                                   "takes_title": 0, "same_as_played": 0,
                                   "accepts": 0, "counters": 0, "robs_leader": 0,
                                   "agrees_with_bot": 0, "usd": 0.0})
        a = r["answer"]
        t["usd"] += r.get("usd") or 0
        if "choice" not in a:
            t["unusable"] += 1
            continue
        t["n"] += 1
        for k in ("takes_win", "buys_dev", "takes_title", "same_as_played", "agrees_with_bot",
                  "accepts", "counters", "robs_leader"):
            t[k] += bool(a.get(k))
    return table


def print_summary(table):
    metric = {"missed_win": "takes_win", "dev_buy": "buys_dev", "control": "same_as_played",
              "road_title": "takes_title", "leader_offer": "accepts",
              "offer_control": "accepts", "robber": "robs_leader"}
    print(f"\n{'set':13s} {'condition':13s} {'answers':>7s}  {'metric':16s} {'rate':>6s}"
          f"  {'= bot':>6s}  {'counters':>8s}  {'unusable':>8s}  {'$':>7s}")
    for (s, c), t in sorted(table.items()):
        m = metric[s]
        rate = f"{t[m] / t['n']:.0%}" if t["n"] else "-"
        bot = f"{t['agrees_with_bot'] / t['n']:.0%}" if t["n"] else "-"
        print(f"{s:13s} {c:13s} {t['n']:7d}  {m:16s} {rate:>6s}  {bot:>6s}  "
              f"{t['counters']:8d}  {t['unusable']:8d}  {t['usd']:7.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--samples", type=int, default=3, help="answers per position and condition")
    ap.add_argument("--conditions", default="as_played,current")
    ap.add_argument("--sets", default="missed_win,dev_buy,control")
    ap.add_argument("--controls", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="scripted answers, spends nothing")
    ap.add_argument("--measure", action="store_true",
                    help="one call per condition on one position, then stop")
    ap.add_argument("--model", default=None,
                    help="ask this model instead of the one that played the positions")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    conditions = args.conditions.split(",")
    sets = args.sets.split(",")
    for path in args.runs:
        first = json.loads(path.open().readline())
        if not args.dry_run and any("dry-run" in (s.get("model") or "") for s in first["seats"]):
            ap.error(f"{path} is a dry run: its agents were scripted, not a model")

    positions = find_positions(args.runs, controls=args.controls)
    for s in list(positions):
        if s not in sets:
            del positions[s]
    print("positions: " + ", ".join(f"{s} {len(v)}" for s, v in positions.items()))

    jobs = []
    for s, items in positions.items():
        for ply, bot in items:
            for c in conditions:
                for k in range(args.samples):
                    jobs.append((s, ply, bot, c, k))
    if args.measure:
        first = next(iter(v for v in positions.values() if v))[0]
        s = next(k for k, v in positions.items() if v)
        jobs = [(s, first[0], first[1], c, 0) for c in conditions]

    if not args.dry_run:
        load_env()
    deciders = {}
    for c in conditions:
        if args.dry_run:
            deciders[c] = ScriptedDecider(heuristic_script("trader", rng=random.Random(1)),
                                          label="dry-run")
        else:
            model = args.model
            for s, items in positions.items():
                if items and model is None:
                    model = items[0][0].seat["model"]
            # One sample per position never reuses a prompt: the saboteur check
            # (2026-10-08) wrote 1.26M tokens to the cache at 1.25x and read
            # 3,768 back. At three samples, re-asks read 66-68% from it.
            deciders[c] = make_decider(model, effort=CONDITIONS[c]["effort"],
                                       cache_writes=args.samples > 1,
                                       **CONDITIONS[c].get("budget", {}))

    OUT_DIR.mkdir(exist_ok=True)
    tag = "dryrun-" if args.dry_run else ("measure-" if args.measure else "")
    out = args.out or OUT_DIR / f"reask-{tag}{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    lock = threading.Lock()
    rows = []
    with out.open("a", buffering=1) as fh:
        plan = {"kind": "plan", "runs": [str(p) for p in args.runs], "conditions": conditions,
                "model": args.model,
                "samples": args.samples, "jobs": len(jobs),
                "positions": {s: [f"{p.start['game_id']}:{p.index}" for p, _ in v]
                              for s, v in positions.items()}}
        fh.write(json.dumps(plan) + "\n")
        print(f"{len(jobs)} calls -> {out}")

        def run(job):
            s, ply, bot, c, k = job
            answer = ask(deciders[c], ply, bot, s, c)
            model = args.model or ply.seat["model"]
            row = {"kind": "answer", "set": s, "condition": c, "sample": k, "model": model,
                   "position": f"{ply.start['game_id']}:{ply.index}",
                   "turn": ply.game.state.num_turns, "color": ply.action.color.value,
                   "persona": ply.seat["name"], "played": describe_action(ply.action),
                   "bot": describe_action(bot), "answer": answer,
                   "usd": None if args.dry_run else cost(answer.get("usage"), model),
                   "logged_input_tokens": (ply.actor.get("usage") or {}).get("input_tokens")
                   if ply.actor.get("attempt") == 0 else None}
            with lock:
                fh.write(json.dumps(row, default=str) + "\n")
                rows.append(row)
                done = len(rows)
            flag = "unusable" if "unusable" in answer else "error" if "error" in answer else ""
            print(f"  {done}/{len(jobs)} {s} {c} {flag}".rstrip(), flush=True)
            return row

        with ThreadPoolExecutor(max_workers=1 if args.dry_run else args.workers) as pool:
            for f in as_completed([pool.submit(run, j) for j in jobs]):
                f.result()

        table = summarize(rows)
        fh.write(json.dumps({"kind": "summary",
                             "table": {f"{s}|{c}": t for (s, c), t in table.items()}}) + "\n")

    print_summary(table)
    if args.measure:
        for r in rows:
            a = r["answer"]
            u = a.get("usage") or {}
            print(f"\n{r['condition']}: {u.get('input_tokens')} in (match logged "
                  f"{r['logged_input_tokens']}), {u.get('output_tokens')} out, "
                  f"{a.get('seconds')}s, ${r['usd'] or 0:.5f}\n  -> {a.get('described')}")
    total = sum(r["usd"] or 0 for r in rows)
    if total:
        print(f"\nspent ${total:.4f}")
    print(f"answers: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

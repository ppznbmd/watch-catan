"""Run one match from the terminal and narrate it.

Seats are `model:persona` pairs, so different models can sit at the same table:

    --seat deepseek-flash:"Hard bargainer" --seat deepseek-v4-pro:"Cooperator"

Repeat a persona across two models to compare the models, or repeat a model
across two personas to compare the prompts. Both are one flag.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

from catanatron.game import TURNS_LIMIT
from catanatron.models.player import Color
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from arena.deciders import DEFAULT_MODEL, ScriptedDecider, make_decider, resolve
from arena.env import DEFAULT_PATH, load_env
from arena.personas import ROSTER, seat_colors
from arena.bot import TradingValuePlayer
from arena.runner import bot_seat, llm_seat, run_match

# $ per million tokens: (input cache miss, input cache hit, output).
# OpenAI: developers.openai.com/api/docs/models/<id>
# DeepSeek: api-docs.deepseek.com/quick_start/pricing/ — PEAK rate here, off-peak
# is half, so a real bill lands at or below what this prints.
# A model that is not listed prints no dollar figure, never a wrong one.
PRICES = {
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-5.6-sol": (4.00, 0.40, 20.00),
    "deepseek-flash": (0.30, 0.006, 1.20),
    "deepseek-v4-pro": (1.32, 0.044, 3.96),
    "claude-opus-5": (5.00, 0.50, 25.00),
    "claude-sonnet-5": (2.00, 0.20, 10.00),
    "claude-haiku-4-5": (1.00, 0.10, 5.00),
}

# Cache writes bill at this multiple of the miss rate. GPT-5.6 writes every
# uncached prompt unless told otherwise (developers.openai.com/api/docs/guides/
# prompt-caching); on 2026-09-18 the dashboard's luna cache writes, $2.244,
# matched every logged uncached luna input token at $0.25/M. Logs from before
# writes were recorded carry no count, and are priced as fully written.
CACHE_WRITE_FACTOR = {"gpt-5.6-": 1.25}
# Flex bills every line, cache writes included, at half the standard rate
# (the GPT-5.6 rows of developers.openai.com/api/docs/pricing, 2026-09-24).
# Applied only when the response reported the flex tier.
FLEX_FACTOR = {"gpt-5.6-": 0.5}
# DeepSeek bills half outside 01:00-04:00 and 06:00-10:00 UTC on weekdays
# (api-docs.deepseek.com/quick_start/pricing, 2026-09-24). A record without the
# time of its request is priced at peak, the most it can have cost. Chinese
# public holidays are off-peak too and are not modelled, so a run on one is
# priced at peak.
OFF_PEAK_FACTOR = {"deepseek-": 0.5}
PEAK_HOURS_UTC = ((1, 4), (6, 10))


def off_peak(requested_at) -> bool:
    """Was a request made at epoch `requested_at` in DeepSeek's off-peak hours?"""
    t = datetime.fromtimestamp(requested_at, timezone.utc)
    return t.weekday() >= 5 or not any(a <= t.hour < b for a, b in PEAK_HOURS_UTC)


def usd(usage, model):
    """Dollars for one usage record, or None for a model with no price on file."""
    if model not in PRICES or not usage:
        return None
    p_miss, p_hit, p_out = PRICES[model]
    hit = usage.get("cached_input_tokens", 0)
    miss = max(usage.get("input_tokens", 0) - hit, 0)
    factor = next((f for prefix, f in CACHE_WRITE_FACTOR.items()
                   if model.startswith(prefix)), 1.0)
    written = min(usage.get("cache_write_input_tokens",
                            miss if factor != 1.0 else 0), miss)
    tier = next((f for prefix, f in FLEX_FACTOR.items() if model.startswith(prefix)), 1.0) \
        if usage.get("service_tier") == "flex" else 1.0
    if usage.get("requested_at") and off_peak(usage["requested_at"]):
        tier *= next((f for prefix, f in OFF_PEAK_FACTOR.items()
                      if model.startswith(prefix)), 1.0)
    return tier * ((miss - written) * p_miss + written * p_miss * factor
                   + hit * p_hit + usage.get("output_tokens", 0) * p_out) / 1e6

# "value" answers trade offers on their merits (arena/bot.py); "value-stock" is
# Catanatron's own, which rejects every offer by tie-break. Matches before prompt
# version 3 were played against the stock bot.
BOTS = {"value": TradingValuePlayer, "value-stock": ValueFunctionPlayer,
        "weighted": WeightedRandomPlayer}

DRY_RUN_STYLES = {
    "Hard bargainer": "tough",
    "Cooperator": "trader",
    "Quiet builder": "quiet",
    "Plain": "trader",
    "Saboteur": "tough",
}


def parse_seat(text):
    """'deepseek-flash:Hard bargainer' -> ('deepseek-flash', 'Hard bargainer').
    A bare persona name uses --model."""
    if ":" in text:
        head, _, tail = text.rpartition(":")
        if tail.strip() in ROSTER:
            return head.strip(), tail.strip()
    return None, text.strip()


def narrate(event):
    kind = event["kind"]
    if kind == "game_start":
        for seat in event["seats"]:
            tag = seat["model"] or seat["kind"]
            print(f"  {seat['color']:7s} {seat['name']:18s} {tag}")
        print()
    elif kind == "bot":
        print(f"  {event['color']:7s} {event['action']['described']}")
    elif kind in ("decision", "fallback", "malformed", "error"):
        color = event.get("color", "?")
        if kind == "decision":
            print(f"  {color:7s} {event['action']['described']}")
            if event.get("say"):
                print(f"          \"{event['say']}\"")
        elif kind == "malformed":
            print(f"  {color:7s} [unusable reply: {event['complaint']}]")
        elif kind == "error":
            print(f"  {color:7s} [call failed: {event['detail']}]")
        else:
            print(f"  {color:7s} [fell back] {event['action']['described']}")


def cost_report(summary):
    lines, total = [], 0.0
    unpriced = set()
    for color, a in summary["agents"].items():
        model = a.get("model")
        usage = a.get("usage") or {}
        tin = usage.get("input_tokens", 0)
        hit = usage.get("cached_input_tokens", 0)
        miss = max(tin - hit, 0)
        tout = usage.get("output_tokens", 0)
        if not tin and not tout:
            continue
        counts = f"{miss:>7,} in {hit:>7,} cached {tout:>7,} out"
        if "cache_write_input_tokens" in usage:
            counts += f" ({usage['cache_write_input_tokens']:,} written to cache)"
        dollars = usd(usage, model)
        if dollars is not None:
            total += dollars
            lines.append(f"  {color:7s} {model:20s} {counts}  ${dollars:.4f}")
        else:
            unpriced.add(model)
            lines.append(f"  {color:7s} {model:20s} {counts}  (no price on file)")
    if not lines:
        return ""
    out = "\n".join(lines)
    if total:
        out += f"\n  {'':7s} {'total':20s} {'':>32}  ${total:.4f}"
    if unpriced:
        out += f"\n  (add {', '.join(sorted(unpriced))} to PRICES in arena/play.py for a dollar figure)"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run one watch-catan match.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # the default model, three personas
  python -m arena.play --bot value

  # another model, no code change
  python -m arena.play --model deepseek-flash --bot value
  WATCH_CATAN_MODEL=claude-haiku-4-5 python -m arena.play --bot value

  # two models at the same table, same persona, to compare the models
  python -m arena.play \\
      --seat gpt-5.6-luna:"Hard bargainer" \\
      --seat deepseek-flash:"Hard bargainer" \\
      --seat gpt-5.6-luna:"Cooperator" --bot value

  # no credentials, nothing spent
  python -m arena.play --dry-run --bot value
""")
    ap.add_argument("--seat", action="append", default=[], metavar="MODEL:PERSONA",
                    help="one seat; repeat up to 4. MODEL may be omitted to use --model")
    ap.add_argument("--bot", action="append", default=[], choices=list(BOTS),
                    help="add a baseline bot seat; repeatable")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"model for seats that name none (default: {DEFAULT_MODEL})")
    ap.add_argument("--effort", default=None,
                    help="reasoning effort passed to the provider. Left unset, the "
                         "provider's own default applies — on gpt-5.6-luna that is "
                         "'medium', and reasoning tokens bill as output. "
                         "'none' is the cheapest setting.")
    ap.add_argument("--flex", action="store_true",
                    help="OpenAI's flex tier: half price, served when there is "
                         "capacity. A call refused through every wait becomes a "
                         "reflex move and is counted in fallbacks")
    ap.add_argument("--max-turns", type=int, default=TURNS_LIMIT,
                    help="end the match with no winner after this many turns "
                         f"(default: the engine's {TURNS_LIMIT})")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", type=Path, default=None, metavar="RUN",
                    help="continue an interrupted match from the roll that opens its "
                         "last logged turn (or --resume-turn), in a new run file. "
                         "Give the same seats it was played with")
    ap.add_argument("--resume-turn", type=int, default=None)
    ap.add_argument("--max-offers", type=int, default=2, help="offers per player per turn")
    ap.add_argument("--dry-run", action="store_true",
                    help="scripted agents instead of a model — spends nothing")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    for complaint in load_env():
        print(f"warning: {DEFAULT_PATH.name}: {complaint}", file=sys.stderr)

    if args.seat:
        specs = [parse_seat(s) for s in args.seat]
    else:
        specs = [(None, n) for n in ("Hard bargainer", "Cooperator", "Quiet builder")]

    unknown = [p for _, p in specs if p not in ROSTER]
    if unknown:
        ap.error(f"unknown persona(s): {', '.join(unknown)}. Known: {', '.join(ROSTER)}")
    if not 2 <= len(specs) + len(args.bot) <= 4:
        ap.error("Catan seats between 2 and 4 players")

    # Fail on a bad model name or a missing key before the board is even built.
    persona_colors, bot_colors = seat_colors([p for _, p in specs], bots=len(args.bot))
    deciders, seats = {}, []
    for i, (model, persona) in enumerate(specs):
        model = model or args.model
        if args.dry_run:
            import random
            from arena.scripted import heuristic_script
            decider = ScriptedDecider(
                heuristic_script(DRY_RUN_STYLES.get(persona, "trader"),
                                 rng=random.Random((args.seed or 0) + i)),
                label=f"dry-run:{model}")
        else:
            try:
                resolve(model)
            except ValueError as exc:
                ap.error(str(exc))
            if model not in deciders:
                try:
                    # 'high' reasons past the default 4,096-token cap, and a
                    # cut-off answer becomes a fallback move (5 of 166 in the
                    # win probe, 2026-10-07).
                    budget = ({"max_tokens": 16384, "timeout": 300.0}
                              if args.effort == "high" else {})
                    if args.flex:
                        # flex keeps its own 15-minute timeout, set in make_decider
                        budget.pop("timeout", None)
                    deciders[model] = make_decider(model, effort=args.effort,
                                                   cache_writes=False, flex=args.flex,
                                                   **budget)
                except (RuntimeError, ValueError) as exc:
                    ap.error(str(exc))
            decider = deciders[model]
        seats.append((Color[persona_colors[i]], llm_seat(ROSTER[persona], decider)))
    for color, bot in zip(bot_colors, args.bot):
        seats.append((Color[color], bot_seat(BOTS[bot])))

    if args.dry_run:
        print("\nwatch-catan — dry run\n")
    else:
        models = sorted({m or args.model for m, _ in specs})
        effort = args.effort or "provider default"
        tier = ", flex" if args.flex else ""
        print(f"\nwatch-catan — {', '.join(models)} (effort: {effort}{tier})\n")
    resume = resume_seed = None
    if args.resume:
        import random
        from arena.rebuild import resume_point
        resume = resume_point(args.resume, args.resume_turn)
        resume_seed = random.randrange(2 ** 31)
        print(f"resuming {args.resume.name} at turn {resume.game.state.num_turns} "
              f"(ply {resume.index}), new dice seed {resume_seed}")
    summary = run_match(seats=seats, seed=args.seed, max_offers_per_turn=args.max_offers,
                        resume=resume, resume_seed=resume_seed,
                        on_event=None if args.quiet else narrate,
                        scratch=args.dry_run, max_turns=args.max_turns)

    print(f"\n{'-' * 72}")
    print(f"winner: {summary['winner'] or 'nobody (turn limit)'} in {summary['turns']} turns "
          f"({summary['seconds']}s, {summary['offers']} offers made)")
    for color, a in summary["agents"].items():
        tag = a.get("model") or "bot"
        saved = a.get("self_offers_skipped", 0)
        extra = f"  ({saved} calls saved)" if saved else ""
        print(f"  {color:7s} {a['name']:18s} {tag:20s} {a['victory_points']:2d} VP  "
              f"{a['llm_calls']:3d} calls  {a['malformed']} unusable  "
              f"{a['fallbacks']} fallbacks{extra}")
    report = cost_report(summary)
    if report:
        print(f"\n{report}")
    print(f"\nreplay: {summary['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

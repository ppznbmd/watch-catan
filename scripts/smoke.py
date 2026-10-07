"""One real call, with a real prompt from a real game.

Answers, before committing to a full match: does the integration work, what does
the model actually write, how long does a decision take, and what does it cost.
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catanatron.game import Game  # noqa: E402
from catanatron.models.actions import generate_playable_actions  # noqa: E402
from catanatron.models.player import Color, RandomPlayer  # noqa: E402

from arena.colors import FULL_TABLE  # noqa: E402
from arena.deciders import DEFAULT_MODEL, make_decider  # noqa: E402
from arena.env import load_env  # noqa: E402
from arena.llm_player import stable_order  # noqa: E402
from arena.personas import ROSTER  # noqa: E402
from arena.play import usd as price  # noqa: E402
from arena.prompt import build_prompt  # noqa: E402
from arena.table_talk import TableTalk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--persona", default="Hard bargainer")
    ap.add_argument("--plies", type=int, default=60, help="how far into a game to go")
    args = ap.parse_args()
    load_env()
    model = args.model or DEFAULT_MODEL

    random.seed(5)
    game = Game(players=[RandomPlayer(c) for c in list(FULL_TABLE)[:3]], seed=5)
    for _ in range(args.plies):
        game.play_tick()

    me = game.state.current_color()
    actions = stable_order(generate_playable_actions(game.state))
    talk = TableTalk()
    talk.record_pitch(game.state.num_turns, "BLUE", (0, 1, 0, 0, 0), (0, 0, 0, 0, 1),
                      "brick for ore — you are not using that ore")
    talk.record(game.state.num_turns, "ORANGE", "reply", "I need the ore. pass.")
    persona = ROSTER[args.persona]
    user = build_prompt(game, me, actions, talk, may_offer=True)

    print(f"model   : {model}")
    print(f"effort  : {args.effort or 'provider default'}")
    print(f"persona : {persona.name}")
    print(f"seat    : {me.value}, turn {game.state.num_turns}, "
          f"{len(actions)} legal moves\n")

    decider = make_decider(model, effort=args.effort)
    started = time.time()
    decision, usage = decider(persona.system_prompt, user)
    elapsed = time.time() - started

    print(f"--- answered in {elapsed:.1f}s ---\n")
    print(f"reasoning: {decision.reasoning}\n")
    chosen = (
        f"authored an offer: give {decision.offer_give} want {decision.offer_want}"
        if decision.choice == -1
        else f"[{decision.choice}] {actions[decision.choice].action_type.value} "
             f"{actions[decision.choice].value}"
        if 0 <= decision.choice < len(actions)
        else f"ILLEGAL index {decision.choice}"
    )
    print(f"choice   : {chosen}")
    print(f"say      : {decision.say or '(nothing)'}\n")

    tin = usage["input_tokens"]
    hit = usage.get("cached_input_tokens", 0)
    tout = usage["output_tokens"]
    written = usage.get("cache_write_input_tokens")
    print(f"tokens   : {tin - hit:,} in, {hit:,} cached, {tout:,} out"
          + (f", {written:,} written to cache" if written is not None else ""))
    usd = price(usage, model)
    if usd is not None:
        print(f"cost     : ${usd:.5f} for this one call")
        print(f"           ${usd * 264:.2f} extrapolated to a 264-call match")
        print(f"           {elapsed * 264 / 60:.0f} min extrapolated, if calls stay serial")


if __name__ == "__main__":
    main()

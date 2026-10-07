"""Report what happened to every trade offer in one or more matches.

    .venv/bin/python scripts/offers.py runs/20260917-192535-214757.jsonl
    .venv/bin/python scripts/offers.py --turns runs/<one>.jsonl

Read-only: it never writes, moves or deletes anything under `runs/`. Pass files
explicitly; it does not glob for them, because `runs/scratch/` holds dry runs
whose scripted agents would be counted as if they were models.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena.events import read_run  # noqa: E402
from arena.offers import mind_changes, offers, steps, summarize  # noqa: E402

COLUMNS = [
    ("offers", "offers"),
    ("confirmed", "traded"),
    ("unanswerable", "nobody could"),
    ("repeated", "repeated"),
    ("repeated_unanswerable", "  ..to nobody"),
    ("sweetened", "sweetened"),
    ("hardened", "hardened"),
    ("reshaped", "reshaped"),
    ("revised_after_a_chosen_no", "revised after no"),
    ("revision_accepted", "  ..accepted"),
    ("withdrawn_after_accept", "withdrawn"),
    ("counters_received", "countered"),
    ("counter_taken_up", "took a counter"),
    ("counter_taken_up_confirmed", "  ..traded"),
]

LEGEND = """\
  nobody could       no seat with a choice was asked: everyone else either could not
                     pay (forced to reject) or was the bot
  repeated           the next offer, same turn, had identical terms
  ..to nobody        of those, repeats of an offer nobody could accept
  sweetened          gives more or asks less of the same resources; hardened: the reverse
  reshaped           different resources
  revised after no   changed terms after a model chose to reject
  ..accepted         and a model then chose to accept the revision
  withdrawn          the proposer cancelled an offer someone had accepted
  countered          counter-offers made to this proposer's offers
  took a counter     offers that put a counter's terms back to its author alone
  ..traded           and closed"""


def print_turns(all_offers):
    by_turn = {}
    for o in all_offers:
        by_turn.setdefault((o.turn, o.proposer), []).append(o)
    for (turn, proposer), seq in by_turn.items():
        print(f"\nturn {turn}  {proposer} ({seq[0].persona})")
        for o in seq:
            answers = "  ".join(f"{c}:{a}" + ("" if h == "chose" else f"({h})")
                                for c, (a, h) in o.answers.items())
            outcome = o.closed or "-"
            if o.partner:
                outcome += f" with {o.partner}"
            print(f"  {o.describe():34s} {answers}")
            print(f"  {'':34s} -> {outcome}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--turns", action="store_true", help="list every offer, turn by turn")
    args = ap.parse_args(argv)

    all_offers = []
    for path in args.runs:
        events = read_run(path)
        if not events or events[0]["kind"] != "game_start":
            ap.error(f"{path}: not a run file (no game_start)")
        found = offers(events)
        finished = events[-1]["kind"] == "game_over"
        print(f"{path.name}: {len(found)} offers" + ("" if finished else "  (match still in progress)"))
        if args.turns:
            print_turns(found)
            print()
        all_offers.extend(found)

    rows = summarize(all_offers)
    if not rows:
        print("no offers")
        return 0
    width = max(len(p) for p in rows)
    print(f"\n{'proposer':{width}s}  " + "  ".join(h for _, h in COLUMNS))
    for persona, r in sorted(rows.items()):
        cells = "  ".join(f"{r.get(k, 0):>{len(h)}d}" for k, h in COLUMNS)
        print(f"{persona:{width}s}  {cells}")
    print(f"\n{LEGEND}")

    flips = mind_changes(all_offers)
    print(f"\nchanged their answer to an identical offer in the same turn: {len(flips)}")
    for first, later, color in flips:
        print(f"  turn {first.turn}: {color} rejected then accepted "
              f"{first.proposer}'s {first.describe()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

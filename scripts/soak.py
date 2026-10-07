"""Prove the harness before spending a token on it.

Random players, many games, every run validated and then deleted — the point is
that nothing crashes and every JSONL is well-formed, not to keep the games.
"""

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catanatron.models.player import Color, RandomPlayer  # noqa: E402

from arena.colors import FULL_TABLE  # noqa: E402
from arena.events import read_run  # noqa: E402
from arena.runner import bot_seat, run_match  # noqa: E402


def validate(path) -> int:
    events = read_run(path)
    assert events[0]["kind"] == "game_start", "run does not open with game_start"
    assert events[-1]["kind"] == "game_over", "run does not close with game_over"
    assert [e["seq"] for e in events] == list(range(len(events))), "sequence has a hole"
    return len(events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--games", type=int, default=100)
    ap.add_argument("--seats", type=int, default=4)
    ap.add_argument("--keep", action="store_true", help="do not delete the runs")
    args = ap.parse_args()

    started = time.time()
    turns, events, sizes, winners = [], [], [], {}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for i in range(args.games):
            summary = run_match(
                seats=[(c, bot_seat(RandomPlayer)) for c in list(FULL_TABLE)[: args.seats]],
                runs_dir=out,
                seed=10_000 + i,
            )
            path = Path(summary["path"])
            events.append(validate(path))
            sizes.append(path.stat().st_size)
            turns.append(summary["turns"])
            winners[summary["winner"]] = winners.get(summary["winner"], 0) + 1
            if not args.keep:
                path.unlink()
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{args.games} games, {time.time() - started:.0f}s elapsed",
                      flush=True)

    elapsed = time.time() - started
    print(f"\n{args.games} games in {elapsed:.1f}s ({elapsed / args.games:.2f}s each)")
    print(f"turns:  median {statistics.median(turns):.0f}, max {max(turns)}")
    print(f"events: median {statistics.median(events):.0f}, max {max(events)}")
    print(f"jsonl:  median {statistics.median(sizes) / 1e6:.1f} MB, "
          f"max {max(sizes) / 1e6:.1f} MB")
    print(f"winners: {winners}")
    print("\nevery run opened with game_start, closed with game_over, "
          "and had an unbroken sequence.")


if __name__ == "__main__":
    main()

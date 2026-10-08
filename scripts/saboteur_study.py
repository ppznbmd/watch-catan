"""Play saboteur pairs back to back under a spending cap.

Each pair is one seed played by {4 Plain} and by {3 Plain + Saboteur}, every
seat on Luna at effort 'high' (scripts/saboteur_pair.sh plays one pair by
hand). Pairs run until the cap would be crossed: before each pair, what has been
spent since `--since` plus two matches at the dearest match so far must fit. A
pair is never started that the cap could cut in half, because half a pair is
not a comparison.

Spending is read back from the run files, not kept in this process, so a
restart after a crash or a reboot picks up the same total.

    .venv/bin/python scripts/saboteur_study.py --since 20261008-0215 --budget 6 \\
        --seeds 202,203,204 --flex
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arena.events import read_run_lenient  # noqa: E402
from arena.play import usd  # noqa: E402

RUNS = ROOT / "runs"
MATCH = r"^[^ ]*python[^ ]* -m arena[.]play"


def match_running() -> bool:
    return subprocess.run(["pgrep", "-f", MATCH], capture_output=True).returncode == 0


def match_cost(path: Path) -> float:
    """Dollars, priced call by call: the summary a match prints prices summed
    usage and so cannot see that a call was served at flex."""
    total = 0.0
    for event in read_run_lenient(path):
        usage = event.get("usage")
        if event.get("kind") != "game_over" and usage and usage.get("model"):
            total += usd(usage, usage["model"]) or 0.0
    return total


def match_line(path: Path) -> str:
    """One line per match for the batch log: what it cost and how long it took,
    read from the run file so a resumed match can be added to its first part."""
    events = read_run_lenient(path)
    states = [e for e in events if e.get("kind") == "state"]
    over = next((e for e in events if e.get("kind") == "game_over"), None)
    minutes = (states[-1]["at"] - events[0]["at"]) / 60 if states else 0
    winner = (over["winner"] or "nobody") if over else "unfinished"
    turns = states[-1]["turn"] if states else 0
    return (f"{path.name}: {winner} at turn {turns}, {minutes:.0f} min, "
            f"${match_cost(path):.3f}")


def study_matches(since: str):
    """Real matches started at or after `since`, a game_id prefix."""
    out = []
    for path in sorted(RUNS.glob("*.jsonl")):
        if path.name[:13] < since:
            continue
        start = read_run_lenient(path)[0]
        if any(s.get("model") and not s["model"].startswith("dry-run") for s in start["seats"]):
            out.append(path)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", required=True,
                    help="game_id prefix (YYYYMMDD-HHMM) from which spending counts")
    ap.add_argument("--budget", type=float, required=True, help="dollars")
    ap.add_argument("--seeds", required=True, help="comma-separated, one pair each; SEED:Saboteur or SEED:Plain plays "
                         "only that half, to complete a pair a crash cut")
    ap.add_argument("--flex", action="store_true")
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--after", type=Path, default=None,
                    help="a saboteur_pair.sh log to wait on until it says 'pair done': "
                         "between its two matches nothing is running, and a batch "
                         "started then would play alongside its second")
    ap.add_argument("--resume", type=Path, default=None,
                    help="an interrupted saboteur match to finish first, from the roll "
                         "that opens its last logged turn")
    ap.add_argument("--first-guess", type=float, default=1.0,
                    help="dollars per match assumed before any has finished")
    args = ap.parse_args(argv)

    log = RUNS / f"batch-saboteur-study-{time.strftime('%Y%m%d-%H%M')}.log"

    def say(text):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {text}"
        print(line, flush=True)
        with log.open("a") as fh:
            fh.write(line + "\n")

    if args.after and "pair done" not in args.after.read_text():
        say(f"waiting for {args.after.name}")
        while "pair done" not in args.after.read_text():
            time.sleep(120)
    if match_running():
        say("waiting: a match is still running")
        while match_running():
            time.sleep(60)

    if args.resume:
        say(f"resume {args.resume.name}; its first part: {match_line(args.resume)}")
        cmd = [str(ROOT / ".venv/bin/python"), "-m", "arena.play",
               "--model", "gpt-5.6-luna", "--effort", "high",
               "--max-turns", str(args.max_turns), "--quiet", "--resume", str(args.resume),
               # the seats it was played with, read back: a cut match may be either half
               *[a for s in read_run_lenient(args.resume)[0]["seats"] for a in ("--seat", s["name"])]]
        if args.flex:
            cmd.append("--flex")
        with log.open("a") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            say(f"FAILED resuming {args.resume.name}, exit {rc}; stopping")
            return rc
        newest = max(RUNS.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        say(f"MATCH {match_line(newest)} (resumed; first part above)")

    for spec in args.seeds.split(","):
        seed, _, half = spec.partition(":")
        seed = int(seed)
        halves = (half,) if half else ("Plain", "Saboteur")
        done = study_matches(args.since)
        costs = [match_cost(p) for p in done]
        spent = sum(costs)
        dearest = max(costs + [args.first_guess])
        if spent + len(halves) * dearest > args.budget:
            say(f"STOP before seed {seed}: spent ${spent:.2f}, it may cost "
                f"${len(halves) * dearest:.2f}, cap ${args.budget:.2f}")
            break
        say(f"seed {seed}: spent so far ${spent:.2f} over {len(done)} matches")
        for fourth in halves:
            say(f"start seed {seed}, fourth seat {fourth}")
            cmd = [str(ROOT / ".venv/bin/python"), "-m", "arena.play",
                   "--model", "gpt-5.6-luna", "--effort", "high",
                   "--max-turns", str(args.max_turns), "--seed", str(seed), "--quiet",
                   "--seat", "Plain", "--seat", "Plain", "--seat", "Plain", "--seat", fourth]
            if args.flex:
                cmd.append("--flex")
            with log.open("a") as fh:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                say(f"FAILED seed {seed}, fourth seat {fourth}, exit {rc}; stopping")
                return rc
            newest = max(RUNS.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
            say(f"done seed {seed}, fourth seat {fourth}")
            say(f"MATCH {match_line(newest)}")
    costs = [match_cost(p) for p in study_matches(args.since)]
    say(f"BATCH OVER: ${sum(costs):.2f} over {len(costs)} matches since {args.since}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

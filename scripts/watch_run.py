"""Follow a live match and emit one line per milestone, for the monitor.

Coverage matters more than tidiness here: if the match crashed or stalled, this
must say so. Silence has to mean "healthy", never "died quietly".
"""
import json, os, sys, time
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"
STALL_SECONDS = 180

def newest():
    files = sorted(RUNS.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None

def alive():
    return os.system("pgrep -f 'arena.play' >/dev/null 2>&1") == 0

path = newest()
while path is None:
    time.sleep(2)
    path = newest()
print(f"following {path.name}", flush=True)

seen, turn, reported, last_line = set(), 0, 0, time.time()
offers = confirms = malformed = fallbacks = 0
fh = path.open()
while True:
    line = fh.readline()
    if not line:
        if time.time() - last_line > STALL_SECONDS:
            print(f"STALLED — no new event for {STALL_SECONDS}s at turn {turn}", flush=True)
            sys.exit(1)
        if not alive():
            print(f"PROCESS GONE at turn {turn} without a game_over — check runs/live.log",
                  flush=True)
            sys.exit(1)
        time.sleep(1)
        continue
    last_line = time.time()
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        continue
    kind = e.get("kind")
    action = (e.get("action") or {}).get("type")

    if kind == "state":
        t = e.get("turn", 0)
        # Compared against the last *reported* turn: comparing against `turn`,
        # which advances on every state event, meant this never fired.
        if t >= reported + 15:
            reported = t
            print(f"turn {t} — {offers} offers, {confirms} trades closed, "
                  f"{malformed} unusable, {fallbacks} fallbacks", flush=True)
        turn = max(turn, t)
    elif action == "OFFER_TRADE":
        offers += 1
        if "offer" not in seen:
            seen.add("offer")
            print(f"FIRST OFFER (turn {e.get('turn', turn)}) {e['color']} "
                  f"{e['action']['described']} — \"{e.get('say','')}\"", flush=True)
    elif action == "CONFIRM_TRADE":
        confirms += 1
        if "confirm" not in seen:
            seen.add("confirm")
            print(f"FIRST TRADE CLOSED — {e['color']} {e['action']['described']}", flush=True)
    elif kind == "malformed":
        malformed += 1
        if malformed in (1, 5, 20):
            print(f"UNUSABLE REPLY #{malformed}: {e['complaint']}", flush=True)
    elif kind == "error":
        print(f"API ERROR: {e['detail']}", flush=True)
    elif kind == "fallback":
        fallbacks += 1
        if fallbacks in (1, 10):
            print(f"FELL BACK TO REFLEX #{fallbacks}: {e.get('complaint')}", flush=True)
    elif kind == "game_over":
        print(f"GAME OVER — winner {e.get('winner')} in {e.get('turns')} turns, "
              f"{e.get('offers')} offers, {e.get('seconds')}s", flush=True)
        sys.exit(0)

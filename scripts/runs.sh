#!/usr/bin/env bash
# Inventory of runs/. Read-only by design: there is no delete here, because the
# two matches lost in this project were lost to bulk file commands.
set -euo pipefail
cd "$(dirname "$0")/.."
# Anchor on the interpreter at the start of the command line. `pgrep -f arena.play`
# also matches any shell whose command line merely contains that text — including
# one writing this file — and a false "a match is running" is worse than no check.
if pgrep -f "^[^ ]*python[^ ]* -m arena[.]play" >/dev/null 2>&1; then
  echo "NOTE: a match is running right now — do not touch runs/"
  echo
fi

for f in runs/*.jsonl runs/scratch/*.jsonl; do
  [ -e "$f" ] || continue
  .venv/bin/python - "$f" <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path) as fh:
        start = json.loads(fh.readline())
        last = None
        for line in fh:
            if '"game_over"' in line:
                last = json.loads(line)
except Exception as exc:
    print(f"  {path}  (unreadable: {exc})"); sys.exit()
models = {s.get("model") for s in start.get("seats", []) if s.get("model")}
real = any(m and not m.startswith("dry-run") for m in models)
tag = "REAL" if real else "dry "
if last:
    print(f"  [{tag}] {path}  {last['turns']} turns, {last['offers']} offers, "
          f"{last['seconds']}s, winner {last['winner']}")
else:
    print(f"  [{tag}] {path}  (no game_over — in progress or interrupted)")
PY
done

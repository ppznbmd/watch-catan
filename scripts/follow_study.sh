#!/usr/bin/env bash
# Follows the saboteur study for a Monitor: batch log milestones, plus anything
# wrong in the live match. Kept in the repo because a copy in /tmp was lost to
# a reboot twice in one day. Silence means healthy; it says so when the batch
# and every match are gone.
cd "$(dirname "$0")/.."
SL=$(ls -t runs/batch-saboteur-study-*.log | head -1)
MATCH='^[^ ]*python[^ ]* -m arena[.]play'
STUDY='^[^ ]*python[^ ]* scripts/saboteur_study[.]py'
tail -n0 -F "$SL" 2>/dev/null | grep --line-buffered -E 'MATCH |FAILED|STOP|BATCH OVER|start seed' &
while pgrep -f "$STUDY" >/dev/null || pgrep -f "$MATCH" >/dev/null; do
  if pgrep -f "$MATCH" >/dev/null; then
    f=$(ls -t runs/*.jsonl | head -1)
    if ! grep -q '"game_over"' "$f"; then
      echo "following $(basename "$f")"
      .venv/bin/python scripts/watch_run.py 2>&1 \
        | grep --line-buffered -E "STALLED|PROCESS GONE|API ERROR|UNUSABLE|FELL BACK"
    fi
  fi
  sleep 20
done
sleep 5
echo "STUDY PROCESSES GONE"; tail -3 "$SL"
kill %1 2>/dev/null

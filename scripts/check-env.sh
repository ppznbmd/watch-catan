#!/usr/bin/env bash
# Report which keys are present. Never prints a value — a key that reaches a
# terminal, a log or a chat has to be treated as leaked.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "no .env — cp .env.example .env"; exit 1; }
while IFS='=' read -r key value; do
  case "$key" in ''|\#*) continue ;; esac
  if [ -n "${value//[[:space:]]/}" ]; then
    echo "  $key: set (${#value} chars)"
  else
    echo "  $key: empty"
  fi
done < .env

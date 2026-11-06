#!/usr/bin/env bash
set -u

LOG="${1:-/data/nos_work/runs/dict_reduce/console.log}"
SUMMARY="${2:-/data/nos_work/results/dict_reduce/summary.json}"

echo "== tail log: $LOG =="
tail -n 30 "$LOG" 2>/dev/null || echo "(no log yet)"

echo ""
echo "== summary.json =="
cat "$SUMMARY" 2>/dev/null || echo "(no summary yet)"

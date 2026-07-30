#!/bin/bash
# ─── UTR Translation Efficiency Pipeline V5 — Dashboard Launcher ─────────────
# Usage: bash launch_dashboard.sh /path/to/results [port]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTDIR="${1:?Usage: bash launch_dashboard.sh /path/to/results [port]}"
PORT="${2:-8050}"

echo "Dashboard starting on http://localhost:${PORT}"

python3 "${SCRIPT_DIR}/dashboard.py" \
    --results "${OUTDIR}" \
    --port    "${PORT}" \
    --host    "0.0.0.0"

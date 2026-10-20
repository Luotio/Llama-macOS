#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next via Strata (OpenAI-compatible API).
#
# Usage:
#   local/serve-qwen38-flash-next.sh
#
# Env overrides: HOST PORT NUMA_NODE OPEN_BROWSER STRATA_CONFIG
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$REPO_ROOT/local/strata/run-unsloth-ud-iq4_xs.sh"

[[ -x "$RUNNER" ]] || { echo "error: runner not found or not executable: $RUNNER" >&2; exit 1; }

if [[ $# -gt 0 ]]; then
    echo "note: positional model argument is ignored; model comes from STRATA_CONFIG"
fi

exec "$RUNNER"

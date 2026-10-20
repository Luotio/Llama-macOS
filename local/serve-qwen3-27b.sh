#!/usr/bin/env bash
# Serve Qwen3.8-27B via Strata.
#
# Usage:
#   STRATA_CONFIG=local/strata/strata-qwen3-27b.json local/serve-qwen3-27b.sh
#
# Env overrides: STRATA_CONFIG HOST PORT NUMA_NODE OPEN_BROWSER
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$REPO_ROOT/local/strata/run-unsloth-ud-iq4_xs.sh"
DEFAULT_CONFIG="$REPO_ROOT/local/strata/strata-qwen3-27b.json"

[[ -x "$RUNNER" ]] || { echo "error: runner not found or not executable: $RUNNER" >&2; exit 1; }

if [[ $# -gt 0 ]]; then
    echo "note: positional model argument is ignored; model comes from STRATA_CONFIG"
fi

if [[ -z "${STRATA_CONFIG:-}" ]]; then
    if [[ -f "$DEFAULT_CONFIG" ]]; then
        export STRATA_CONFIG="$DEFAULT_CONFIG"
    else
        echo "error: no Strata config for 27B found."
        echo "set STRATA_CONFIG to a valid file, for example:"
        echo "  STRATA_CONFIG=$DEFAULT_CONFIG local/serve-qwen3-27b.sh"
        exit 1
    fi
fi

exec "$RUNNER"

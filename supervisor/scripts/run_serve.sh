#!/usr/bin/env bash
# Wrapper for llmcli_serve daemon — sources .env, starts llmcli serve in the
# background, blocks until the OpenAI endpoint is healthy, then re-foregrounds.
# Supervisor sees exit 0 only after the engine is actually ready.
#
# Env vars:
#   LLMCLI_PORT        — HTTP port to probe (default: 8091)
#   LLMCLI_PROBE_TIMEOUT — max seconds to wait for readiness (default: 180)

set -euo pipefail

# --- source .env (secrets must never live in supervisor conf) ----------------
set -a
# shellcheck source=/dev/null
[ -f "$HOME/projects/llmCLI/.env" ] && source "$HOME/projects/llmCLI/.env"
set +a

# --- config ------------------------------------------------------------------
PORT="${LLMCLI_PORT:-8091}"
TIMEOUT="${LLMCLI_PROBE_TIMEOUT:-180}"
POLL_INTERVAL=2
PROBE_URL="http://localhost:${PORT}/health"

# --- launch daemon in background ---------------------------------------------
llmcli serve &
SERVE_PID=$!

# --- readiness probe loop ----------------------------------------------------
elapsed=0
ready=0

while [ "$elapsed" -lt "$TIMEOUT" ]; do
    if curl -sf --max-time 2 "$PROBE_URL" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep "$POLL_INTERVAL"
    elapsed=$(( elapsed + POLL_INTERVAL ))
done

if [ "$ready" -eq 0 ]; then
    echo "llmcli readiness probe timed out after ${TIMEOUT}s — killing serve (PID ${SERVE_PID})" >&2
    kill "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
    exit 1
fi

# --- re-foreground: block until daemon exits (supervisor tracks this PID) ----
wait "$SERVE_PID"

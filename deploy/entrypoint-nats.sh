#!/bin/bash
set -e

MODE="${1:-llm}"
shift || true

case "$MODE" in
    llm)
        exec llmcli nats-serve llm "$@"
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        exit 1
        ;;
esac
